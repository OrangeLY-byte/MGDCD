import sys
from os.path import join as opj
from os.path import dirname as opd
import numpy as np
from model.SCE import SCE_single
from utils.eval_auroc import compute_matrix_auroc, mask_top, cosine_similarity, shd_distance

sys.path.append(opj(opd(__file__), ".."))
import tqdm
import torch
from torch import  nn
from utils.utils_tool import gumbel_softmax, spars_loss
from utils.fmriopt_type import CDopt
from model.TCD import Tem_Cau_Decoder

class SDCD_NET(object):
    def __init__(self, args: CDopt.CDargs, device="cuda"):
        self.args = args
        self.device = device
        self.lambda_1 = self.args.encoder.lambda_1
        self.decoder = Tem_Cau_Decoder(self.args.n_nodes, in_ch=self.args.data_dim,n_layers=self.args.decoder.gru_layers,
                                           hidden_ch=self.args.decoder.mlp_hid,shared_weights_decoder=self.args.decoder.shared_weights_decoder,
                                           concat_h=self.args.decoder.concat_h).to(self.device)
        self.encoder = SCE_single(n_nodes=self.args.n_nodes, in_dim=args.input_step, hid_dim=args.encoder.hid_dim).to(self.device)
        self.pred_loss = nn.MSELoss()
        self.spar_loss = spars_loss(mode='js')
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr = self.args.decoder.lr_dec, weight_decay=self.args.decoder.weight_decay)
        self.decoder_scheduler = torch.optim.lr_scheduler.StepLR(self.decoder_optimizer, step_size=50, gamma=0.95)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.args.encoder.lr_enc, weight_decay=5e-4)
        self.encoder_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_optimizer, step_size=50, gamma=0.95)


    def decoder_stage(self, train_batch_data,i):
        def sample_bernoulli(sample_matrix, batch_size):
            sample_matrix = sample_matrix[None].expand(batch_size, -1, -1)
            return torch.bernoulli(sample_matrix).float()
        self.decoder.train()
        self.decoder_optimizer.zero_grad()
        causal = self.encoder(train_batch_data, i)
        batch_data = train_batch_data[i]
        x, y, t = batch_data
        causal = torch.mean(causal, dim=0)
        causal_sampled = sample_bernoulli(causal, self.args.batch_size)
        y_pred = self.decoder(x, causal_sampled)
        loss = self.pred_loss(y, y_pred)
        loss.backward()
        self.decoder_optimizer.step()
        return loss

    def encoder_stage(self, train_batch_data,i):
        def gumbel_sigmoid_sample(graph, batch_size, tau=1.25):
            prob = graph[None, :, :, None].expand(batch_size, -1, -1, -1)
            logits = torch.concat([prob, (1 - prob)], axis=-1)
            samples = gumbel_softmax(logits, tau=tau, hard=True)[:, :, :, 0]
            return samples
        self.encoder.train()
        self.encoder_optimizer.zero_grad()
        causal = self.encoder(train_batch_data,i)
        batch_data = train_batch_data[i]
        x, y, t = batch_data
        causal = torch.mean(causal, dim=0)
        causal_sampled = gumbel_sigmoid_sample(causal, self.args.batch_size)
        y_pred = self.decoder(x, causal_sampled)
        loss_spar = self.spar_loss(causal)
        loss_pre = self.pred_loss(y, y_pred)
        loss = loss_pre + self.lambda_1 * loss_spar
        loss.backward()
        self.encoder_optimizer.step()
        return loss, loss_pre, loss_spar

    def train(self, train_batch_data, test_batch_data, true_causal):
        auroc = 0
        latent_pred_step = 0
        graph_discov_step = 0
        pbar = tqdm.tqdm(total=self.args.total_epoch)
        entity_auroc = []
        entity_shd = []
        entity_cs = []
        for epoch_i in range(self.args.total_epoch):
            # train decoder
            for i,_ in enumerate(train_batch_data):
                latent_pred_step += self.args.batch_size
                loss = self.decoder_stage(train_batch_data,i)
                pbar.set_postfix_str(f"decoder_stage loss={loss.item():.2f},auroc={auroc:.4f}")
            self.decoder_scheduler.step()

           # train encoder
            for i, _ in enumerate(train_batch_data):
                graph_discov_step += self.args.batch_size
                loss, loss_pre, loss_spar = self.encoder_stage(train_batch_data,i)
                pbar.set_postfix_str(f"encoder_stage loss={loss.item():.2f}, pre={loss_pre.item():.2f},spr={loss_spar.item():.2f},auroc={auroc:.4f}")
            self.encoder_scheduler.step()

            pbar.update(1)
            # test
            test_auroc_list = []
            test_shd_list = []
            test_cs_list = []
            self.encoder.eval()
            with torch.no_grad():
                for i, _ in enumerate(test_batch_data):
                    causal = self.encoder(test_batch_data, i)
                    causal = torch.mean(causal, dim=0)
                    true_causal = np.abs(true_causal)
                    causal = causal.detach().numpy()
                    auroc = compute_matrix_auroc(torch.tensor(causal), torch.tensor(true_causal))
                    percent = true_causal.sum() / self.args.n_nodes ** 2
                    GC_Bin_est = mask_top(causal, percent)
                    shd = shd_distance(GC_Bin_est, true_causal)
                    cs = cosine_similarity(torch.tensor(GC_Bin_est), torch.tensor(true_causal), self.args.n_nodes)
                    test_auroc_list.append(auroc)
                    test_shd_list.append(shd)
                    test_cs_list.append(cs)
                test_auroc = np.mean(test_auroc_list)
                test_shd = np.mean(test_shd_list)
                test_cs = np.mean(test_cs_list)
            entity_auroc.append(test_auroc)
            entity_shd.append(test_shd)
            entity_cs.append(test_cs)
        auroc_out = sum(entity_auroc[-5:]) / len(entity_auroc[-5:])
        shd_out = sum(entity_shd[-5:]) / len(entity_shd[-5:])
        cs_out = sum(entity_cs[-5:]) / len(entity_cs[-5:])
        return auroc_out, shd_out, cs_out


def main(train_batch_data,test_batch_data,true_causal,opt, device="cpu"):
    SDCD = SDCD_NET(opt,device=device)
    test_auroc,test_shd,test_cs = SDCD.train(train_batch_data, test_batch_data,true_causal)
    return test_auroc,test_shd,test_cs


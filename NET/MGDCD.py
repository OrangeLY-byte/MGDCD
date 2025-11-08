import sys
from os.path import join as opj
from os.path import dirname as opd
import numpy as np
from model.SCE import Spat_Cau_Encoder
from utils.eval_auroc import cosine_similarity, compute_matrix_auroc, mask_top, shd_distance
sys.path.append(opj(opd(__file__), ".."))
import tqdm
import torch
from torch import  nn
from utils.utils_tool import gumbel_softmax, spars_loss, consistency_loss
from utils.opt_type import MGDCDopt
from model.TCD import Tem_Cau_Decoder


class MGDCD_NET(object):
    def __init__(self, args: MGDCDopt.MGDCDargs, device="cuda"):
        self.args = args
        self.device = device
        self.lambda_1 = self.args.encoder.lambda_1
        self.lambda_2 = self.args.encoder.lambda_2
        self.lambda_3 = self.args.encoder.lambda_3
        self.decoder = nn.ModuleList([Tem_Cau_Decoder(self.args.n_nodes, in_ch=self.args.data_dim,n_layers=self.args.decoder.gru_layers,hidden_ch=self.args.decoder.mlp_hid,shared_weights_decoder=self.args.decoder.shared_weights_decoder,
                                           concat_h=self.args.decoder.concat_h).to(self.device) for _ in range(args.n_sub)])
        self.encoder = Spat_Cau_Encoder(n_sub = self.args.n_sub, n_nodes=self.args.n_nodes, in_dim=args.input_step, hid_dim=args.encoder.hid_dim).to(self.device)
        self.pred_loss = nn.MSELoss()
        self.spar_loss = spars_loss(mode='js')
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr = self.args.decoder.lr_dec, weight_decay=1e-4)
        self.decoder_scheduler = torch.optim.lr_scheduler.StepLR(self.decoder_optimizer, step_size=25, gamma=0.75)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.args.encoder.lr_enc, weight_decay=1e-4)
        self.encoder_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_optimizer, step_size=2, gamma=0.9)


    def decoder_stage(self, train_batch_data,i):
        def sample_bernoulli(sample_matrix, batch_size):
            sample_matrix = sample_matrix[None].expand(batch_size, -1, -1)
            return torch.bernoulli(sample_matrix).float()
        self.decoder.train()
        self.decoder_optimizer.zero_grad()
        loss = 0
        causal_s, causal_c, causal_c_all = self.encoder(train_batch_data, i)
        causal_s, causal_c = torch.sigmoid(causal_s), torch.sigmoid(causal_c)
        for m,_ in enumerate(self.decoder):
            batch_data_j = train_batch_data[m][i]
            x, y, t = batch_data_j
            common_expanded = causal_c
            causal_m = torch.max(common_expanded, causal_s[:, m, :, :])
            causal_m = torch.mean(causal_m, dim=0)
            causal_sampled = sample_bernoulli(causal_m, self.args.batch_size)
            y_pred = self.decoder[m](x,causal_sampled)
            loss_j = self.pred_loss(y, y_pred)
            loss += loss_j
        loss.backward()
        self.decoder_optimizer.step()
        return loss

    def encoder_stage(self, train_batch_data,i):
        def gumbel_sigmoid_sample(graph, batch_size, tau=1.01):
            prob = graph[None, :, :, None].expand(batch_size, -1, -1, -1)
            logits = torch.concat([prob, (1 - prob)], axis=-1)
            samples = gumbel_softmax(logits, tau=tau, hard=True)[:, :, :, 0]
            return samples
        self.encoder.train()
        self.encoder_optimizer.zero_grad()
        loss,loss_pre,loss_spar,loss_con,loss_dec = 0.0, 0.0, 0.0, 0.0,0.0
        causal_s, causal_c, causal_c_all = self.encoder(train_batch_data,i)
        causal_s, causal_c, causal_c_all = torch.sigmoid(causal_s), torch.sigmoid(causal_c), torch.sigmoid(causal_c_all)
        loss_con = consistency_loss(causal_c_all,causal_c.unsqueeze(1),mode='js')
        overlap_graph = torch.mean(causal_s * causal_c.unsqueeze(1),dim=0)
        loss_dec = torch.linalg.norm(overlap_graph.flatten(), ord=1) /(self.args.n_nodes * self.args.n_nodes)
        for m in range(self.args.n_sub):
            batch_data_m = train_batch_data[m][i]
            x, y, t = batch_data_m
            causal_m = torch.max(causal_c, causal_s[:,m,:,:])
            causal_m = torch.mean(causal_m,dim=0)
            causal_sampled = gumbel_sigmoid_sample(causal_m, self.args.batch_size)
            y_pred = self.decoder[m](x, causal_sampled)
            loss_spar_m = self.spar_loss(causal_m)
            loss_pre_m = self.pred_loss(y, y_pred)
            loss_spar += loss_spar_m
            loss_pre += loss_pre_m
        loss = loss_pre + self.lambda_1 * loss_spar + self.lambda_2 * loss_con  + self.lambda_3 * loss_dec
        loss.backward()
        self.encoder_optimizer.step()
        return loss, loss_pre, loss_spar, loss_con ,loss_dec

    def train(self, train_batch_data, test_batch_data, true_causal,true_common):
        auroc = 0
        common_auroc = 0
        latent_pred_step = 0
        graph_discov_step = 0
        pbar = tqdm.tqdm(total=self.args.total_epoch,dynamic_ncols=True)
        common_out = []
        entity_out = []
        common_shd_out = []
        entity_shd_out = []
        common_cs_out = []
        entity_cs_out = []
        for epoch_i in range(self.args.total_epoch):
            # train decoder
            for i,_ in enumerate(train_batch_data[0]):
                latent_pred_step += self.args.batch_size
                loss = self.decoder_stage(train_batch_data,i)
                pbar.set_postfix_str(f"d loss={loss.item():.2f}, aur={auroc:.4f}, c_aur={common_auroc:.4f}")
            self.decoder_scheduler.step()

           # train encoder
            for i, _ in enumerate(train_batch_data[0]):
                graph_discov_step += self.args.batch_size
                loss, loss_pre, loss_spar,loss_con, loss_dec = self.encoder_stage(train_batch_data,i)
                pbar.set_postfix_str(f"e pre={loss_pre.item():.2f},spr={loss_spar.item():.2f},match={loss_con.item():.2f}, ove={loss_dec.item():.2f}, auroc={auroc:.4f}, c_auroc={common_auroc:.4f}")
            self.encoder_scheduler.step()

            pbar.update(1)
            # test
            test_auroc_list = []
            test_common_auroc_list = []
            test_shd_list = []
            test_common_shd_list = []
            test_cs_list = []
            test_common_cs_list = []
            self.encoder.eval()
            with torch.no_grad():
                for i, _ in enumerate(test_batch_data[0]):
                    auroc = 0
                    shd = 0
                    cs = 0
                    causal_s, causal_c, causal_c_all = self.encoder(train_batch_data, i)
                    causal_s, causal_c, causal_c_all = torch.sigmoid(causal_s), torch.sigmoid(causal_c), torch.sigmoid(causal_c_all)
                    true_common = np.abs(true_common)
                    causal_m_all = []
                    for m in range(self.args.n_sub):
                        common_expanded = causal_c
                        causal_m = torch.max(common_expanded, causal_s[:, m, :, :])
                        causal_m = torch.mean(causal_m, dim=0)
                        true_causal_m = np.abs(true_causal[m])
                        causal_m = causal_m.detach().numpy()
                        auroc_m = compute_matrix_auroc(torch.tensor(causal_m), torch.tensor(true_causal_m))
                        percent = true_causal_m.sum() / self.args.n_nodes ** 2
                        causal_bi_m = mask_top(causal_m, percent)
                        cs_m = cosine_similarity(torch.tensor(causal_bi_m), torch.tensor(true_causal_m), self.args.n_nodes)
                        shd_m = shd_distance(causal_bi_m, true_causal_m)
                        auroc += auroc_m
                        shd += shd_m
                        cs += cs_m
                        causal_m_all.append(causal_m)
                    auroc /= 3
                    shd /= 3
                    cs /= 3
                    test_auroc_list.append(auroc)
                    test_shd_list.append(shd)
                    test_cs_list.append(cs)
                    causal_c = torch.mean(causal_c, dim=0)
                    causal_c = causal_c.detach().numpy()
                    causal_c = 0.1 * np.mean(causal_m_all,axis=0) + 0.9 * causal_c
                    common_auroc = compute_matrix_auroc(torch.tensor(causal_c), torch.tensor(true_common))
                    percent_common = true_common.sum() / self.args.n_nodes ** 2
                    causal_bi_common = mask_top(causal_c, percent_common)
                    common_cs = cosine_similarity(torch.tensor(causal_bi_common), torch.tensor(true_common), self.args.n_nodes)
                    common_shd = shd_distance(causal_bi_common,true_common)
                    test_common_auroc_list.append(common_auroc)
                    test_common_shd_list.append(common_shd)
                    test_common_cs_list.append(common_cs)
                test_auroc = np.mean(test_auroc_list)
                test_shd = np.mean(test_shd_list)
                test_cs = np.mean(test_cs_list)
                test_common_auroc = np.mean(test_common_auroc_list)
                test_common_shd = np.mean(test_common_shd_list)
                test_common_cs = np.mean(test_common_cs_list)
            common_out.append(test_common_auroc)
            common_shd_out.append(test_common_shd)
            common_cs_out.append(test_common_cs)
            entity_out.append(test_auroc)
            entity_shd_out.append(test_shd)
            entity_cs_out.append(test_cs)
        return (sum(entity_out[-5:]) / len(entity_out[-5:]), sum(common_out[-5:]) / len(common_out[-5:]), sum(entity_shd_out[-5:]) / len(entity_shd_out[-5:]),
                sum(common_shd_out[-5:]) / len(common_shd_out[-5:]), sum(entity_cs_out[-5:]) / len(entity_cs_out[-5:]),
                sum(common_cs_out[-5:]) / len(common_cs_out[-5:]), )



def main(train_batch_data,test_batch_data,true_causal,true_common, opt, device="cpu"):
    MGDCD = MGDCD_NET(opt,device=device)
    test_auroc, test_common_auroc, test_shd, test_common_shd, test_cs, test_common_cs = MGDCD.train(train_batch_data, test_batch_data,true_causal,true_common)
    return round(test_auroc,4), round(test_common_auroc,4), round(test_shd,4), round(test_common_shd,4), round(test_cs,4), round(test_common_cs,4)


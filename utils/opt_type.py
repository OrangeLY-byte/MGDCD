from dataclasses import dataclass
from typing import Any, List

@dataclass
class ReproducOpt:
    seed: int
    benchmark: bool
    deterministic: bool


@dataclass
class MGDCDopt:

    @dataclass
    class MGDCDargs:
        n_nodes: int
        n_sub: int
        input_step: int
        batch_size: int
        data_dim: int
        total_epoch: int


        @dataclass
        class decoder:
            lr_dec = float
            concat_h = bool
            shared_weights_decoder = bool
            mlp_hid = int
            gru_layers = int
            model: str
            merge_policy: str
            weight_decay: int
            prob: bool

        @dataclass
        class encoder:
            lr_enc = float
            hid_dim = int
            lambda_1 = float
            lambda_2 = float
            lambda_3 = float
            disable_graph: bool
            use_true_graph: bool
    
    reproduc: ReproducOpt
    log: Any
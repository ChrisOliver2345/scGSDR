import sys
from absl import logging
import torch as pt
from utils import get_cfg
from core import core
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train a model with specified parameters")
    parser.add_argument("--refer_dataset_path", type=str, required=True)
    parser.add_argument("--query_dataset_path", type=str, required=True)

    parser.add_argument("--lr", type=float, required=False, default=1e-5, help="Learning rate")
    parser.add_argument("--n_episodes", type=int, required=False, default=300, help="Max epochs")
    parser.add_argument("--tau", type=float, required=False, default=0.5, help="Model hyperparameters")
    parser.add_argument("--ebd_d", type=int, required=False, default=256, help="Model hyperparameters")
    parser.add_argument("--gcn_d", type=int, required=False, default=64, help="Model hyperparameters")
    parser.add_argument("--n_classes", type=int, required=False, default=2, help="Number of classification categories")
    parser.add_argument("--sessions", type=int, required=False, default=1, help="Number of model runs")
    parser.add_argument("--stop_epochs", type=int, required=False, default=20, help="Number of epochs for early stopping")
    parser.add_argument("--early_stop", type=bool, required=False, default=False, help="Enable or disable early stopping")
    parser.add_argument("--refresh", type=int, required=False, default=0, help="Whether to force refresh the pathway scored file")
    parser.add_argument("--hvg", type=int, required=False, default=500, help="Dimensions of the HVG matrix")
    parser.add_argument("--temperature", type=int, required=False, default=100, help="Control the feature fusion function")
    parser.add_argument("--lambda_recon", type=float, required=False, default=1e-5, help="Weight of the graph reconstruction loss")
    parser.add_argument("--init_top_k", type=int, required=False, default=20, help="Initialization of the graph")
    parser.add_argument("--use_edge_attr", type=bool, required=False, default=True, help="Whether to use a weighted graph")
    parser.add_argument("--num_dgc_layers", type=int, required=False, default=3, help="Number of GCN layers used")
    parser.add_argument("--num_encoder_layers", type=int, required=False, default=1, help="Number of encoder layers used")
    parser.add_argument("--num_decoder_layers", type=int, required=False, default=1, help="Number of dncoder layers used")
    parser.add_argument("--lambda_d", type=float, required=False, default=1e-5, help="Domain adaptation loss weight")

    args = parser.parse_args()

    cfg = get_cfg()
    cfg.lr = args.lr
    cfg.n_episodes = args.n_episodes
    cfg.tau = args.tau
    cfg.ebd_d = args.ebd_d
    cfg.gcn_d = args.gcn_d
    cfg.n_classes = args.n_classes
    cfg.sessions = args.sessions
    cfg.stop_epochs = args.stop_epochs
    cfg.early_stop = args.early_stop
    cfg.refresh = args.refresh

    cfg.hvg = args.hvg
    cfg.temperature = args.temperature
    cfg.lambda_recon = args.lambda_recon
    cfg.init_top_k = args.init_top_k
    cfg.use_edge_attr = args.use_edge_attr
    cfg.num_dgc_layers = args.num_dgc_layers
    cfg.num_encoder_layers = args.num_encoder_layers
    cfg.num_decoder_layers = args.num_decoder_layers
    cfg.lambda_d = args.lambda_d
    cfg.refer_dataset_path = args.refer_dataset_path
    cfg.query_dataset_path = args.query_dataset_path

    cfg.max_dim = 0
    cfg.train_num = 0
    cfg.test_num = 0
    cfg.state_dict_path = ''
    cfg.device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
    print("device: "+str(cfg.device))

    logging.set_verbosity(logging.INFO)

    core(cfg)


if __name__ == '__main__':
    main()

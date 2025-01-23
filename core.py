import random
import sys
import numpy as np
import pandas as pd
import torch as pt
from trainer import Trainer
from preprocess import data_loader_all, prepare_data


def core(cfg):

    cfg.t_repetition = 6
    path_r = cfg.refer_dataset_path
    path_q = cfg.query_dataset_path

    (refer_graph_features, refer_label, refer_node_features,
     query_graph_features, query_label, query_node_features) = prepare_data(path_r, path_q, cfg)

    feature_r, labels_r, node_feature_list_r, \
        feature_q, labels_q, node_feature_list_q, \
        feature_t, node_feature_list_t, \
        state_pathway_data, pathway_idx, cfg, \
        = data_loader_all(cfg, refer_graph_features, refer_label, refer_node_features,
                          query_graph_features, query_label, query_node_features)
    for j in range(cfg.sessions):
        trainer = Trainer(cfg, state_pathway_data[1])
        run_trainer(pathway_idx, trainer, cfg, feature_r, labels_r, node_feature_list_r,
                    feature_q, labels_q, node_feature_list_q, state_pathway_data,
                    feature_t, node_feature_list_t)


def run_trainer(pathway_idx, trainer, cfg, feature_r, labels_r, node_feature_list_r,
                feature_q, labels_q, node_feature_list_q, state_pathway_data,
                feature_t=None, node_feature_list_t=None):

    trainer.trainer(pathway_idx=pathway_idx, graph_r=feature_r, node_r=node_feature_list_r, label_r=labels_r,
                    graph_q=feature_q, node_q=node_feature_list_q, label_q=labels_q,
                    graph_t=feature_t, node_t=node_feature_list_t, state_pathway_data=state_pathway_data)


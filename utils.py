import argparse
import sys
from typing import Any, Mapping
import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy.sparse import coo_matrix
from torch_geometric.utils import from_scipy_sparse_matrix
import math


def get_cfg():
    parser = argparse.ArgumentParser(description='')
    cfg = parser.parse_args(args=[])
    return cfg


def construct_graph(x_ebd_q, x_ebd_k, cfg) -> torch.Tensor:
    x_ebd_q = F.softmax(x_ebd_q, dim=-1)
    x_ebd_k = F.softmax(x_ebd_k, dim=-1)

    if x_ebd_q.ndim == 3:
        adjacency_matrix = torch.matmul(x_ebd_q, torch.transpose(x_ebd_k, 1, 2))
    else:
        adjacency_matrix = torch.matmul(x_ebd_q, torch.transpose(x_ebd_k, 0, 1))

    adjacency_matrix = adjacency_matrix / math.sqrt(x_ebd_q.size(-1))

    return adjacency_matrix


def get_coo(adjacency_matrix) -> torch.Tensor:
    adj_matrices = []
    adjacency_matrix = adjacency_matrix.cpu().detach().numpy()
    for i in range(len(adjacency_matrix)):
        coo = coo_matrix(adjacency_matrix[i])
        data = from_scipy_sparse_matrix(coo)
        adj_matrices.append(data)

    return adj_matrices


class SemanticAttention(torch.nn.Module):

    def __init__(self,
                 cfg: Mapping[str, Any]) -> None:
        super().__init__()
        pathway_ebd = int(cfg.tau * cfg.t_repetition * cfg.ebd_d)
        self.spatial_attn = torch.nn.Sequential(
            torch.nn.Linear(in_features=cfg.t_repetition * cfg.ebd_d, out_features=pathway_ebd, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=pathway_ebd, out_features=cfg.t_repetition * cfg.ebd_d, bias=False),
            torch.nn.Sigmoid()
            )
        self.cfg = cfg

    def forward(self, x_ebd):
        x_spatial_attn = x_ebd.reshape(self.cfg.n_neurons, self.cfg.t_repetition * self.cfg.max_dim)
        x_spatial_attn = self.spatial_attn(x_spatial_attn)
        x_spatial_attn = x_spatial_attn.reshape(self.cfg.t_repetition, self.cfg.n_neurons, self.cfg.max_dim)

        return x_spatial_attn


class GraphAttention(torch.nn.Module):

    def __init__(self,
                 cfg: Mapping[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.temporal_attn = torch.nn.Sequential(torch.nn.Linear(cfg.t_repetition, cfg.t_repetition, bias=False),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(cfg.t_repetition, cfg.t_repetition, bias=False),
                                              torch.nn.Sigmoid())

    def forward(self, x_ebd: torch.Tensor):
        x_temporal_attn = x_ebd.view(self.cfg.t_repetition, self.cfg.n_neurons * self.cfg.ebd_d)
        x_temporal_attn = torch.mean(x_temporal_attn, -1)
        x_temporal_attn = self.temporal_attn(x_temporal_attn)
        x_temporal_attn = x_temporal_attn.view(self.cfg.t_repetition, 1, 1)
        return x_temporal_attn


class Sparsify(nn.Module):
    def __init__(self, cfg, scale_factor_init=1.0):
        super().__init__()
        self.top_k = cfg.init_top_k
        self.scale_factor = nn.Parameter(torch.tensor([scale_factor_init]))

    def forward(self, adjacency_matrix):
        flattened_values = adjacency_matrix.view(-1)
        global_threshold = torch.topk(flattened_values, self.top_k, largest=True).values[-1]
        scaled_threshold = global_threshold * self.scale_factor
        sparsified_adjacency = torch.relu(adjacency_matrix - scaled_threshold)

        return sparsified_adjacency


class feature_sparsify(torch.nn.Module):

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        self.threshold = torch.nn.parameter.Parameter(torch.full((1,), -4, dtype=torch.float32))

    def forward(self, matrix: torch.Tensor) -> torch.Tensor:
        sparse_matrix = torch.relu(matrix - torch.sigmoid(self.threshold))
        return sparse_matrix


class UpSamplingNet(torch.nn.Module):
    def __init__(self,
                 cfg: Mapping[str, Any]) -> None:
        super(UpSamplingNet, self).__init__()
        self.fc1 = torch.nn.Linear(14, 64)
        self.fc2 = torch.nn.Linear(64, 200)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x
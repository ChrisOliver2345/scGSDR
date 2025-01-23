import sys
from utils import *
from utils import SemanticAttention, GraphAttention, Sparsify, feature_sparsify, construct_graph, get_coo
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import torch_geometric as tg

class Attention(nn.Module):
    def __init__(self, in_channels, temperature):
        super().__init__()
        self.dense_weight = nn.Linear(in_features=in_channels, out_features=1)
        self.dropout = nn.Dropout(0.1)
        self.temperature = temperature

    def forward(self, stacked):
        logits = self.dense_weight(stacked)
        weights = F.softmax(logits / self.temperature, dim=1)
        outputs = torch.sum(stacked*weights, dim=1)

        return outputs


class GraphLearner(torch.nn.Module):

    def __init__(self, cfg, state_dim):
        super().__init__()
        self.cfg = cfg
        self.dyn_graph_learner = DynGraphLearner(cfg)
        self.dyn_graph_classifier = DynGraphClassifier(cfg, state_dim)
        self.attention_fusion_layer = Attention(self.cfg.gcn_d, self.cfg.temperature)
        self.fc = nn.Linear(in_features=cfg.gcn_d, out_features=2, bias=True)

    def forward(self, x_graph, x_node, state_pathway):
        x_split = torch.stack(
            [x_graph[:, t * self.cfg.max_dim:(t + 1) * self.cfg.max_dim] for t in range(self.cfg.t_repetition)], 0)
        sparse_adjacency, mt_sparse, graphs_ori, mt_attn_pathway = self.dyn_graph_learner(x_split)
        out, mt_attn = self.dyn_graph_classifier(x_node, sparse_adjacency, state_pathway)
        out_fusion = self.attention_fusion_layer(out)
        z = self.fc(out_fusion)

        return z, mt_sparse, out_fusion, graphs_ori, mt_attn, mt_attn_pathway


class GradReverse(Function):

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * -1)

def grad_reverse(x):
    return GradReverse.apply(x)


class Discriminator(nn.Module):
    def __init__(self, dropout_rate, h_dim):
        super(Discriminator, self).__init__()
        self.D1 = torch.nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim // 2),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(h_dim // 2, 1))
        self.Drop1 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # x = grad_reverse(x)
        yhat = self.Drop1(self.D1(x))
        return torch.sigmoid(yhat)



# Dynamic Graph Learner


def reshape(x_save):
    d0 = x_save.shape[0]
    d1 = x_save.shape[1]
    d2 = x_save.shape[2]
    x_save = np.concatenate(x_save, axis=1)
    x_save = x_save.reshape(d1, d2 * d0)
    return x_save

class DynGraphLearner(torch.nn.Module):
    """
    Dynamic graph learner
    """

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        self.semantic_attention_q = SemanticAttention(cfg)
        self.semantic_attention_k = SemanticAttention(cfg)
        self.graph_attention_q = GraphAttention(cfg)
        self.graph_attention_k = GraphAttention(cfg)
        self.graph_sparsify = Sparsify(cfg)
        self.feat_sparsify = feature_sparsify(cfg)
        self.cfg = cfg

    def forward(self, x_split):
        x_ebd_q = x_split
        x_semantic_attention_q = self.semantic_attention_q(x_ebd_q)
        x_ebd_q = x_semantic_attention_q * x_ebd_q
        x_graph_attn_q = self.graph_attention_q(x_ebd_q)
        x_ebd_q = x_graph_attn_q * x_ebd_q

        x_ebd_q_output = x_ebd_q

        x_ebd_k = x_split
        x_semantic_attention_k = self.semantic_attention_k(x_ebd_k)
        x_ebd_k = x_semantic_attention_k * x_ebd_k
        x_graph_attn_k = self.graph_attention_k(x_ebd_k)
        x_ebd_k = x_graph_attn_k * x_ebd_k

        x_ebd_q = self.feat_sparsify(x_ebd_q)
        x_save = x_ebd_q.cpu().detach().numpy()
        x_save = reshape(x_save)
        adjacency_matrix = construct_graph(x_ebd_q, x_ebd_k, self.cfg)
        adjacency_matrix = self.graph_sparsify(adjacency_matrix)
        coo_matrix = get_coo(adjacency_matrix)

        return coo_matrix, x_save, adjacency_matrix, x_ebd_q_output


# Dynamic Graph Classifier


class SamplingNet(torch.nn.Module):
    def __init__(self, cfg, input_dims) -> None:
        super(SamplingNet, self).__init__()
        self.number_of_inputs = cfg.t_repetition
        assert len(input_dims) == self.number_of_inputs, "input_dims length must match cfg.t_repetition"

        self.fc_layers = torch.nn.ModuleList()
        for i in range(self.number_of_inputs):
            self.fc_layers.append(torch.nn.Sequential(
                torch.nn.Linear(input_dims[i], 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256)
            ))

    def forward(self, node_features):
        assert len(node_features) == self.number_of_inputs, "node_features length must match number_of_inputs"

        new_features = []
        for i in range(self.number_of_inputs):
            feature = self.fc_layers[i](node_features[i])
            new_features.append(feature)

        return new_features


class DGCLayer(torch.nn.Module):
    """
    Dynamic graph classifier layer
    """

    def __init__(self, cfg, input_d) -> None:
        super().__init__()
        self.cfg = cfg
        self.number_of_gcn = cfg.t_repetition
        self.gcn_layers = torch.nn.ModuleList([
            tg.nn.GCNConv(in_channels=input_d*(self.number_of_gcn-1), out_channels=cfg.gcn_d, bias=False)
            for _ in range(self.number_of_gcn)
        ])

    def forward(self, gcn_input, sparse_adjacency) -> torch.Tensor:
        outputs = []

        for i in range(self.number_of_gcn):

            combined_input = torch.cat([gcn_input[j] for j in range(self.number_of_gcn) if j != i], dim=-1)
            # combined_input = torch.stack([gcn_input[j] for j in range(self.number_of_gcn) if j != i], dim=0).sum(dim=0)
            # combined_input = torch.stack([gcn_input[j] for j in range(self.number_of_gcn) if j != i], dim=0).mean(dim=0)
            edge_index = sparse_adjacency[i][0].to(self.cfg.device)

            if self.cfg.use_edge_attr:
                edge_attr = torch.unsqueeze(sparse_adjacency[i][1], dim=1).float().to(self.cfg.device)
                out = self.gcn_layers[i](combined_input, edge_index, edge_attr)
            else:
                out = self.gcn_layers[i](combined_input, edge_index)
            outputs.append(out)

        out = torch.stack(outputs, dim=0)
        return out

class DimensionMapper_State(torch.nn.Module):
    def __init__(self, input_dims, output_dims=16):
        super(DimensionMapper_State, self).__init__()
        self.fc = torch.nn.Linear(input_dims, output_dims)

    def forward(self, x):
        x = self.fc(x)
        return x

class StateTransformer(torch.nn.Module):
    def __init__(self, cfg, num_genes=16, d_model=16):
        super().__init__()
        self.cfg = cfg
        self.embeddings = torch.nn.ModuleList([torch.nn.Linear(in_features=num_genes, out_features=d_model) for _ in range(14)])
        self.transformer = torch.nn.Transformer(d_model=d_model, nhead=2, num_encoder_layers=self.cfg.num_encoder_layers, num_decoder_layers=self.cfg.num_decoder_layers)
        self.fc_out = torch.nn.Linear(in_features=d_model*14, out_features=self.cfg.gcn_d)
        self.features_out_hook = []

    def hook(self, module, fea_in, fea_out):
        self.features_out_hook.clear()
        self.features_out_hook.append(fea_out)
        return None

    def forward(self, src):
        src = torch.stack([self.embeddings[i](src[:, i, :]) for i in range(14)], dim=1)
        src = src.permute(1, 0, 2)
        output = self.transformer(src, src)
        output = output.permute(1, 0, 2)
        output = output.reshape(output.size(0), -1)
        output = self.fc_out(output)

        get_attention = True
        if get_attention:

            layer_name = 'encoder.layers.0.self_attn'
            for (name, module) in self.transformer.named_modules():
                if name == layer_name:
                    module.register_forward_hook(hook=self.hook)

            return output, self.features_out_hook

        return output, -1

class DynGraphClassifier(torch.nn.Module):
    """
    Dynamic graph classifier
    """

    def __init__(self, cfg, state_pathway_dims):
        super().__init__()
        self.cfg = cfg

        self.dyn_graph_cls = torch.nn.ModuleList([SamplingNet(cfg, cfg.gene_num)])
        input_d = 256
        for _ in range(cfg.num_dgc_layers):
            self.dyn_graph_cls.append(DGCLayer(cfg, input_d))
            input_d = cfg.gcn_d

        self.mapper = torch.nn.ModuleList([
            DimensionMapper_State(input_dims) for input_dims in state_pathway_dims
        ])
        self.StateTransformer = StateTransformer(cfg)

    def forward(self, node_features, sparse_adjacency, state_pathway):
        for idx, layer in enumerate(self.dyn_graph_cls):
            if idx == 0:  # SamplingNet
                node_features = layer(node_features)
            else:  # DGCLayers
                node_features = layer(node_features, sparse_adjacency)

        state_matrices = []
        for mt, mapper in zip(state_pathway, self.mapper):
            mt = mapper(mt)
            state_matrices.append(mt)

        state_matrices = torch.stack(state_matrices, dim=1)
        state_features, mt_attn = self.StateTransformer(state_matrices)

        if len(mt_attn) >= 1:
            mt_attn0 = mt_attn[0]
            mt_attn0 = mt_attn0[0]
        else:
            mt_attn0 = mt_attn

        state_features = state_features.unsqueeze(0)
        out = torch.cat((node_features, state_features), dim=0)
        out = out.permute(1, 0, 2)

        return out, mt_attn0

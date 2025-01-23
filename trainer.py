import os, sys
from absl import logging
import torch
import torch.nn as nn
from model import GraphLearner, Discriminator
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
import torch.nn.functional as F

logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)

class Trainer():

    def __init__(self, cfg, state_dim):
        self.cfg = cfg
        self.cfg.loss = "ce"
        self.model = GraphLearner(cfg, state_dim).to(cfg.device)
        if os.path.exists(cfg.state_dict_path):
            self.model.load_state_dict(torch.load(cfg.state_dict_path, map_location=f'{cfg.device}'))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.discriminator = Discriminator(0, cfg.gcn_d).to(cfg.device)
        self.optimizer_discr = torch.optim.Adam(self.discriminator.parameters(), lr=cfg.lr * 0.01)

    def cal_score(self, target, pred, test):
        prob = F.softmax(pred, dim=1)[:, 1]
        prob_np = prob.cpu().detach().numpy()
        target_auc = target[:, 1]
        target_auc_np = target_auc.cpu().detach().numpy()

        roc_auc = roc_auc_score(y_true=target_auc_np, y_score=prob_np)
        aupr = average_precision_score(target_auc_np, prob_np)

        pred = F.softmax(pred, dim=1)
        _, pred = torch.max(pred, 1)
        _, target = torch.max(target, 1)
        pred_np = pred.cpu().detach().numpy()
        target_np = target.cpu().detach().numpy()

        acc = accuracy_score(y_true=target_np, y_pred=pred_np)
        f1_micro = f1_score(y_true=target_np, y_pred=pred_np, average="micro")
        f1_macro = f1_score(y_true=target_np, y_pred=pred_np, average="macro")
        f1_weighted = f1_score(y_true=target_np, y_pred=pred_np, average="weighted")

        score_list = [roc_auc, aupr, acc, f1_micro, f1_macro, f1_weighted]
        return score_list


    def train(self, x_graph, x_node, target, state_pathway, t_graph=None, t_node=None, t_state_pathway=None):

        self.optimizer.zero_grad()
        if t_graph is not None:
            self.cfg.n_neurons = self.cfg.train_num
            pred_response_r, mt_sparse_r, embd_r, graphs_r, _, _ = self.model(x_graph, x_node, state_pathway)
            self.cfg.n_neurons = self.cfg.transfer_num
            _, _, embd_t, _, _, _ = self.model(t_graph, t_node, t_state_pathway)

            x_discr = torch.cat((embd_r, embd_t), dim=0)
            pred_discr = self.discriminator(x_discr)
            label_source = torch.ones(embd_r.size(0), 1)
            label_target = torch.zeros(embd_t.size(0), 1)
            label_discr = torch.cat([label_source, label_target], 0)

            loss_r, loss_discr, score_list_r = self.func_loss(pred_response=pred_response_r,
                                                  target=target,
                                                  test=0,
                                                  graphs=[graphs_r, embd_r],
                                                  pred_discr=pred_discr,
                                                  label_discr=label_discr
                                                  )

        else:
            self.cfg.n_neurons = self.cfg.train_num
            pred_response_r, mt_sparse_r, embd_r, graphs_r, _, _ = self.model(x_graph, x_node, state_pathway)

            loss_r, score_list_r = self.func_loss(pred_response=pred_response_r,
                                                  target=target,
                                                  test=0,
                                                  graphs=[graphs_r, embd_r]
                                                  )

        loss_r.backward()
        self.optimizer.step()

        return loss_r, score_list_r, mt_sparse_r, embd_r

    def train_discr(self, x_graph, x_node, target, state_pathway, t_graph, t_node, t_state_pathway):
        self.optimizer_discr.zero_grad()
        self.cfg.n_neurons = self.cfg.train_num
        pred_response_r, mt_sparse_r, embd_r, graphs_r, _, _ = self.model(x_graph, x_node, state_pathway)
        self.cfg.n_neurons = self.cfg.transfer_num
        _, _, embd_t, _, _, _ = self.model(t_graph, t_node, t_state_pathway)

        x_discr = torch.cat((embd_r, embd_t), dim=0)
        pred_discr = self.discriminator(x_discr)
        label_source = torch.ones(embd_r.size(0), 1)
        label_target = torch.zeros(embd_t.size(0), 1)
        label_discr = torch.cat([label_source, label_target], 0)

        f_loss_bce = nn.BCELoss(reduction='mean')
        loss_discr = f_loss_bce(pred_discr, label_discr.to(self.cfg.device))

        loss_discr.backward()
        self.optimizer_discr.step()

        return loss_discr

    def func_loss(self, pred_response, target, test, graphs, pred_discr=None, label_discr=None):
        target = target
        pred_response = pred_response

        score_list = self.cal_score(target, pred_response, test)

        if test == 1:
            return torch.tensor(-1), score_list

        _, TARGET_indices = torch.max(target, 1)

        if self.cfg.loss == 'ce':
            f_loss_ce = nn.CrossEntropyLoss()
            loss_ce = f_loss_ce(pred_response, TARGET_indices)
            loss_fin = loss_ce

        graphs_ori = graphs[0].to(self.cfg.device)
        graphs_ori = graphs_ori.sum(dim=0)
        feat = graphs[1].to(self.cfg.device)
        graph_recon = torch.matmul(feat, feat.t())
        threshold = torch.topk(graph_recon.view(-1), self.cfg.init_top_k, largest=True).values[-1]
        graph_recon = torch.relu(graph_recon - threshold)
        loss_recon = F.mse_loss(graph_recon, graphs_ori)

        if pred_discr is not None:
            f_loss_bce = nn.BCELoss(reduction='mean')
            loss_DG = f_loss_bce(pred_discr, label_discr.to(self.cfg.device))
            loss_fin = loss_fin + self.cfg.lambda_recon*loss_recon - self.cfg.lambda_d * loss_DG
            return loss_fin, loss_DG, score_list

        loss_fin = loss_fin + self.cfg.lambda_recon * loss_recon

        return loss_fin, score_list

    def test(self, x_graph, x_node, target, state_pathway):
        with torch.no_grad():
            pred_response_q, mt_sparse_q, embd_q, graphs_q, mt_attn_q, attn_pathway_q = self.model(x_graph, x_node, state_pathway)

            mt_attn_q = mt_attn_q.permute(1, 0, 2)
            mt_attn_q = torch.sum(mt_attn_q, dim=2)

            loss_q, score_list_q = self.func_loss(pred_response=pred_response_q, target=target, test=1, graphs=[graphs_q, embd_q])

            pred_response_q = F.softmax(pred_response_q, dim=1)
            _, pred_response_q = torch.max(pred_response_q, 1)

        return loss_q, score_list_q, mt_sparse_q, embd_q, mt_attn_q, attn_pathway_q, pred_response_q


    def trainer(self, pathway_idx, graph_r, node_r, label_r, graph_q, node_q, label_q, state_pathway_data, graph_t=None, node_t=None):

        state_pathway_r = state_pathway_data[2]
        if graph_t is not None:
            state_pathway_t = state_pathway_data[4]

        for t in range(self.cfg.n_episodes):
            if graph_t is not None:
                self.model.train()
                self.discriminator.train()
                self.cfg.n_neurons = self.cfg.train_num
                if t % 5 < 4:
                    loss_train, scores_train, mt_sparse_train, embd_train = self.train(x_graph=graph_r,
                                                                                       x_node=node_r,
                                                                                       target=label_r,
                                                                                       state_pathway=state_pathway_r,
                                                                                       t_graph=graph_t,
                                                                                       t_node=node_t,
                                                                                       t_state_pathway=state_pathway_t
                                                                                       )
                else:
                    loss_discr = self.train_discr(x_graph=graph_r,
                                                  x_node=node_r,
                                                  target=label_r,
                                                  state_pathway=state_pathway_r,
                                                  t_graph=graph_t,
                                                  t_node=node_t,
                                                  t_state_pathway=state_pathway_t
                                                  )

            else:
                self.model.train()
                self.cfg.n_neurons = self.cfg.train_num
                loss_train, scores_train, mt_sparse_train, embd_train = self.train(x_graph=graph_r,
                                                                                   x_node=node_r,
                                                                                   target=label_r,
                                                                                   state_pathway=state_pathway_r
                                                                                   )

            if t % 5 <= 4:
                logging.info(
                    f'iteration: {t:04d} '
                    f'--train loss: {loss_train.item():.5f} '
                    f'-- train AUC: {scores_train[0].item():.4f} '
                    f'-- train ACC: {scores_train[2].item():.4f} '
                    )





import sys
import numpy as np
import pandas as pd
import torch
from r_code import get_mat_path
import scanpy as sc
import os
import anndata
import csv
from torch.nn.functional import pad

def connect_features(tensor_list):
    dim = max(tensor.shape[1] for tensor in tensor_list)
    print(dim)
    expanded_tensors = []
    for tensor in tensor_list:
        padding_size = dim - tensor.shape[1]
        if padding_size > 0:
            padded_tensor = pad(tensor, (0, padding_size), "constant", 0)
            expanded_tensors.append(padded_tensor)
        else:
            expanded_tensors.append(tensor)
    graph_features = torch.cat(expanded_tensors, dim=1)
    return graph_features, dim, dim


def lower_matrix(df):
    df = df.T
    index = df.index
    index = list(index)
    index2 = []
    for x in index:
        index2.append(x.lower())
    df.index = index2
    return df

def lower_matrix_inner(df):
    index = df.columns
    index = list(index)
    index2 = []
    for x in index:
        index2.append(x.lower())
    df.columns = index2
    return df

def common_path(r_M, q_M):
    r_index = r_M.index
    q_index = q_M.index
    common_path = r_index.intersection(q_index)
    r_M = r_M.loc[common_path]
    q_M = q_M.loc[common_path]
    r_M = r_M.T
    q_M = q_M.T
    pathway_dim = r_M.shape[1]
    return r_M, q_M, pathway_dim


def max_value(*args):
    if not args:
        return None
    max_val = args[0]
    for num in args[1:]:
        if num > max_val:
            max_val = num
    return max_val


def expand_dataframe(df, max_dim):
    current_dim = df.shape[1]

    if current_dim < max_dim:
        num_cols_to_add = max_dim - current_dim
        cols_to_add = pd.DataFrame(0, index=df.index, columns=range(current_dim, current_dim + num_cols_to_add))
        df = pd.concat([df, cols_to_add], axis=1)

    return df


def cut_label(query_label, transfer_idx, test_idx):
    labels_transfer = query_label[transfer_idx]
    labels_test = query_label[test_idx]
    return labels_transfer, labels_test

def pathway_fliter(cfg, train, test, transfer_idx=None):
    pathway_Metabolism = "./Pathways/Human_Metabolism.txt"
    pathway_Cellular_Process = "./Pathways/Human_Cellular_Process.txt"
    pathway_Environmental_Information_Processing = "./Pathways/Human_Environmental_Information_Processing.txt"
    pathway_Organismal_Systems = "./Pathways/Human_Organismal_Systems.txt"
    pathway_Disease = "./Pathways/Human_Disease.txt"
    pathway_Drug = "./Pathways/Human_PharmGKB_Drug.txt"
    pathway_state = "./Pathways/Cell_states.txt"

    train = lower_matrix_inner(train)
    test = lower_matrix_inner(test)

    Metabolism_set = set()
    Cellular_Process_set = set()
    Environmental_Information_Processing_set = set()
    Organismal_Systems_set = set()
    Disease_set = set()
    Drug_set = set()

    def update_set_from_file(file_path, gene_set):
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                genes = row[2:]
                gene_set.update(genes)

    update_set_from_file(pathway_Metabolism, Metabolism_set)
    update_set_from_file(pathway_Cellular_Process, Cellular_Process_set)
    update_set_from_file(pathway_Environmental_Information_Processing, Environmental_Information_Processing_set)
    update_set_from_file(pathway_Organismal_Systems, Organismal_Systems_set)
    update_set_from_file(pathway_Disease, Disease_set)
    update_set_from_file(pathway_Drug, Drug_set)

    state_list = []
    state_name = []
    with open(pathway_state, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            name = row[0]
            genes = row[2:]
            state_name.append(name)
            state_list.append(genes)

    train_state_list = []
    state_dim = []
    for genes in state_list:
        filtered_df = train.filter(items=genes)
        dims_num = filtered_df.shape[1]
        state_dim.append(dims_num)
        train_state_list.append(filtered_df)

    train_Metabolism = train.filter(items=Metabolism_set)
    train_Cellular_Process = train.filter(items=Cellular_Process_set)
    train_Environmental_Information_Processing = train.filter(items=Environmental_Information_Processing_set)
    train_Organismal_Systems = train.filter(items=Organismal_Systems_set)
    train_Disease = train.filter(items=Disease_set)
    train_Drug = train.filter(items=Drug_set)

    num = [train_Metabolism.shape[1],
           train_Cellular_Process.shape[1],
           train_Environmental_Information_Processing.shape[1],
           train_Organismal_Systems.shape[1],
           train_Disease.shape[1],
           train_Drug.shape[1]
           ]

    test_state_list = []
    transfer_state_list = []
    for genes in state_list:
        filtered_df = test.filter(items=genes)
        test_state_list.append(filtered_df)
        if transfer_idx is not None:
            transfer_df = filtered_df.loc[transfer_idx]
            transfer_state_list.append(transfer_df)

    test_Metabolism = test.filter(items=Metabolism_set)
    test_Cellular_Process = test.filter(items=Cellular_Process_set)
    test_Environmental_Information_Processing = test.filter(items=Environmental_Information_Processing_set)
    test_Organismal_Systems = test.filter(items=Organismal_Systems_set)
    test_Disease = test.filter(items=Disease_set)
    test_Drug = test.filter(items=Drug_set)


    if transfer_idx is not None:
        transfer_Metabolism = test_Metabolism.loc[transfer_idx]
        transfer_Cellular_Process = test_Cellular_Process.loc[transfer_idx]
        transfer_Environmental_Information_Processing = test_Environmental_Information_Processing.loc[transfer_idx]
        transfer_Organismal_Systems = test_Organismal_Systems.loc[transfer_idx]
        transfer_Disease = test_Disease.loc[transfer_idx]
        transfer_Drug = test_Drug.loc[transfer_idx]

        transfer_Metabolism = torch.FloatTensor(np.array(transfer_Metabolism))
        transfer_Cellular_Process = torch.FloatTensor(np.array(transfer_Cellular_Process))
        transfer_Environmental_Information_Processing = torch.FloatTensor(np.array(transfer_Environmental_Information_Processing))
        transfer_Organismal_Systems = torch.FloatTensor(np.array(transfer_Organismal_Systems))
        transfer_Disease = torch.FloatTensor(np.array(transfer_Disease))
        transfer_Drug = torch.FloatTensor(np.array(transfer_Drug))

        transfer_list = [transfer_Metabolism, transfer_Cellular_Process, transfer_Environmental_Information_Processing,
                         transfer_Organismal_Systems, transfer_Disease, transfer_Drug]


    train_Metabolism = torch.FloatTensor(np.array(train_Metabolism))
    train_Cellular_Process = torch.FloatTensor(np.array(train_Cellular_Process))
    train_Environmental_Information_Processing = torch.FloatTensor(np.array(train_Environmental_Information_Processing))
    train_Organismal_Systems = torch.FloatTensor(np.array(train_Organismal_Systems))
    train_Disease = torch.FloatTensor(np.array(train_Disease))
    train_Drug = torch.FloatTensor(np.array(train_Drug))

    train_list = [train_Metabolism, train_Cellular_Process, train_Environmental_Information_Processing,
                  train_Organismal_Systems, train_Disease, train_Drug]

    test_Metabolism = torch.FloatTensor(np.array(test_Metabolism))
    test_Cellular_Process = torch.FloatTensor(np.array(test_Cellular_Process))
    test_Environmental_Information_Processing = torch.FloatTensor(np.array(test_Environmental_Information_Processing))
    test_Organismal_Systems = torch.FloatTensor(np.array(test_Organismal_Systems))
    test_Disease = torch.FloatTensor(np.array(test_Disease))
    test_Drug = torch.FloatTensor(np.array(test_Drug))
    test_list = [test_Metabolism, test_Cellular_Process, test_Environmental_Information_Processing,
                  test_Organismal_Systems, test_Disease, test_Drug]

    if transfer_idx is not None:
        return train_list, test_list, num, train_state_list, test_state_list, state_name, state_dim, transfer_list, transfer_state_list
    else:
        return train_list, test_list, num, train_state_list, test_state_list, state_name, state_dim


def data_loader_all(cfg, refer_graph_features, refer_label, refer_node_features, query_graph_features, query_label, query_node_features):

    num_sample = int(query_graph_features.shape[0] * 0.3)
    transfer_idx = np.random.choice(query_graph_features.index, size=num_sample, replace=False)
    transfer_nidx = [query_graph_features.index.get_loc(name) for name in transfer_idx]
    cfg.transfer_num = num_sample

    query_num = query_node_features.shape[0]
    refer_num = refer_node_features.shape[0]
    cfg.train_num = refer_num
    cfg.test_num = query_num

    refer_node_features_list, query_node_features_list, dims, train_state_list, test_state_list, state_name, state_dim, transfer_node_features_list, transfer_state_list = \
        pathway_fliter(cfg, refer_node_features, query_node_features, transfer_idx)

    cfg.gene_num = dims

    refer_label = refer_label['response'].values
    query_label = query_label['response'].values
    refer_label = np.vstack([1 - refer_label, refer_label]).T
    query_label = np.vstack([1 - query_label, query_label]).T

    refer_graph_features = lower_matrix(refer_graph_features)
    query_graph_features = lower_matrix(query_graph_features)

    r_path = "./Temp/features_r_temp.csv"
    q_path = "./Temp/features_q_temp.csv"
    refer_graph_features.to_csv(r_path, sep=',')
    query_graph_features.to_csv(q_path, sep=',')

    pa_r_Metabolism = "./Temp/refer_Metabolism.csv"
    pa_q_Metabolism = "./Temp/query_Metabolism.csv"
    pa_r_Cellular_Process = "./Temp/refer_Cellular_Process.csv"
    pa_q_Cellular_Process = "./Temp/query_Cellular_Process.csv"
    pa_r_Environmental_Information_Processing = "./Temp/refer_Environmental_Information_Processing.csv"
    pa_q_Environmental_Information_Processing = "./Temp/query_Environmental_Information_Processing.csv"
    pa_r_Organismal_Systems = "./Temp/refer_Organismal_Systems.csv"
    pa_q_Organismal_Systems = "./Temp/query_Organismal_Systems.csv"
    pa_r_Disease = "./Temp/refer_Disease.csv"
    pa_q_Disease = "./Temp/query_Disease.csv"
    pa_r_Drug = "./Temp/refer_Drug.csv"
    pa_q_Drug = "./Temp/query_Drug.csv"

    get_mat_path(r_path, cfg, [pa_r_Metabolism, pa_r_Cellular_Process, pa_r_Environmental_Information_Processing,
                               pa_r_Organismal_Systems, pa_r_Disease, pa_r_Drug])
    get_mat_path(q_path, cfg, [pa_q_Metabolism, pa_q_Cellular_Process, pa_q_Environmental_Information_Processing,
                               pa_q_Organismal_Systems, pa_q_Disease, pa_q_Drug])

    r_M_Metabolism = pd.read_csv(pa_r_Metabolism, index_col=0, header=0)
    r_M_Cellular_Process = pd.read_csv(pa_r_Cellular_Process, index_col=0, header=0)
    r_M_Environmental_Information_Processing = pd.read_csv(pa_r_Environmental_Information_Processing, index_col=0,header=0)
    r_M_Organismal_Systems = pd.read_csv(pa_r_Organismal_Systems, index_col=0, header=0)
    r_M_Disease = pd.read_csv(pa_r_Disease, index_col=0, header=0)
    r_M_Drug = pd.read_csv(pa_r_Drug, index_col=0, header=0)

    q_M_Metabolism = pd.read_csv(pa_q_Metabolism, index_col=0, header=0)
    q_M_Cellular_Process = pd.read_csv(pa_q_Cellular_Process, index_col=0, header=0)
    q_M_Environmental_Information_Processing = pd.read_csv(pa_q_Environmental_Information_Processing, index_col=0, header=0)
    q_M_Organismal_Systems = pd.read_csv(pa_q_Organismal_Systems, index_col=0, header=0)
    q_M_Disease = pd.read_csv(pa_q_Disease, index_col=0, header=0)
    q_M_Drug = pd.read_csv(pa_q_Drug, index_col=0, header=0)

    r_M_Metabolism, q_M_Metabolism, dim_Metabolism = common_path(r_M_Metabolism, q_M_Metabolism)
    r_M_Cellular_Process, q_M_Cellular_Process, dim_Cellular_Process = common_path(r_M_Cellular_Process, q_M_Cellular_Process)
    r_M_Environmental_Information_Processing, q_M_Environmental_Information_Processing, dim_Environmental_Information_Processing = common_path(r_M_Environmental_Information_Processing, q_M_Environmental_Information_Processing)
    r_M_Organismal_Systems, q_M_Organismal_Systems, dim_Organismal_Systems = common_path(r_M_Organismal_Systems, q_M_Organismal_Systems)
    r_M_Disease, q_M_Disease, dim_Disease = common_path(r_M_Disease, q_M_Disease)
    r_M_Drug, q_M_Drug, dim_Drug = common_path(r_M_Drug, q_M_Drug)

    max_dim = max_value(dim_Metabolism, dim_Cellular_Process, dim_Environmental_Information_Processing, dim_Organismal_Systems, dim_Disease, dim_Drug)

    cfg.max_dim = max_dim
    cfg.ebd_d = max_dim
    r_M_Metabolism = expand_dataframe(r_M_Metabolism, max_dim)
    q_M_Metabolism = expand_dataframe(q_M_Metabolism, max_dim)
    r_M_Cellular_Process = expand_dataframe(r_M_Cellular_Process, max_dim)
    q_M_Cellular_Process = expand_dataframe(q_M_Cellular_Process, max_dim)
    r_M_Environmental_Information_Processing = expand_dataframe(r_M_Environmental_Information_Processing, max_dim)
    q_M_Environmental_Information_Processing = expand_dataframe(q_M_Environmental_Information_Processing, max_dim)
    r_M_Organismal_Systems = expand_dataframe(r_M_Organismal_Systems, max_dim)
    q_M_Organismal_Systems = expand_dataframe(q_M_Organismal_Systems, max_dim)
    r_M_Disease = expand_dataframe(r_M_Disease, max_dim)
    q_M_Disease = expand_dataframe(q_M_Disease, max_dim)
    r_M_Drug = expand_dataframe(r_M_Drug, max_dim)
    q_M_Drug = expand_dataframe(q_M_Drug, max_dim)

    t_M_Metabolism = q_M_Metabolism.iloc[transfer_nidx]
    t_M_Cellular_Process = q_M_Cellular_Process.iloc[transfer_nidx]
    t_M_Environmental_Information_Processing = q_M_Environmental_Information_Processing.iloc[transfer_nidx]
    t_M_Organismal_Systems = q_M_Organismal_Systems.iloc[transfer_nidx]
    t_M_Disease = q_M_Disease.iloc[transfer_nidx]
    t_M_Drug = q_M_Drug.iloc[transfer_nidx]

    transfer_graph_features = pd.concat(
        [t_M_Metabolism, t_M_Cellular_Process, t_M_Environmental_Information_Processing, t_M_Organismal_Systems,
         t_M_Disease, t_M_Drug], axis=1)

    transfer_graph_features = torch.FloatTensor(np.array(transfer_graph_features))
    transfer_graph_features = transfer_graph_features.to(cfg.device)

    transfer_node_features_list = [feature.to(cfg.device) for feature in transfer_node_features_list]
    for i in range(len(transfer_state_list)):
        transfer_state_list[i] = torch.tensor(transfer_state_list[i].values, dtype=torch.float32).to(cfg.device)


    refer_graph_features = pd.concat(
        [r_M_Metabolism, r_M_Cellular_Process, r_M_Environmental_Information_Processing, r_M_Organismal_Systems,
         r_M_Disease, r_M_Drug], axis=1)
    refer_graph_features = torch.FloatTensor(np.array(refer_graph_features))
    refer_label = torch.FloatTensor(refer_label)

    query_graph_features = pd.concat(
        [q_M_Metabolism, q_M_Cellular_Process, q_M_Environmental_Information_Processing, q_M_Organismal_Systems,
         q_M_Disease, q_M_Drug], axis=1)

    pathway_idx_all = query_graph_features.index
    pathway_col_all = query_graph_features.columns
    pathway_idx = [pathway_idx_all, pathway_col_all]

    query_graph_features = torch.FloatTensor(np.array(query_graph_features))
    query_label = torch.FloatTensor(query_label)

    refer_graph_features = refer_graph_features.to(cfg.device)
    refer_label = refer_label.to(cfg.device)

    refer_node_features_list = [feature.to(cfg.device) for feature in refer_node_features_list]

    query_graph_features = query_graph_features.to(cfg.device)
    query_label = query_label.to(cfg.device)

    query_node_features_list = [feature.to(cfg.device) for feature in query_node_features_list]

    for i in range(len(train_state_list)):
        train_state_list[i] = torch.tensor(train_state_list[i].values, dtype=torch.float32).to(cfg.device)

    for i in range(len(test_state_list)):
        test_state_list[i] = torch.tensor(test_state_list[i].values, dtype=torch.float32).to(cfg.device)

    return (refer_graph_features, refer_label, refer_node_features_list,
            query_graph_features, query_label, query_node_features_list,
            transfer_graph_features, transfer_node_features_list,
            [state_name, state_dim, train_state_list, test_state_list, transfer_state_list], pathway_idx, cfg)


def prepare_data(path_r, path_q, cfg):
    if os.path.exists(path_r) and os.path.exists(path_q):
        print('File paths are valid.')
    else:
        print('File paths are invalid. Please check paths.')
        print(path_r)
        print(path_q)
        sys.exit()

    r = pd.read_table(path_r, delimiter=',', index_col=0, header=0)
    q = pd.read_table(path_q, delimiter=',', index_col=0, header=0)

    train_L = pd.DataFrame(r.iloc[:, 0])
    train_M = r.iloc[:, 1:]
    test_L = pd.DataFrame(q.iloc[:, 0])
    test_M = q.iloc[:, 1:]

    if cfg.hvg != -1:
        adata_train = sc.AnnData(train_M)
        adata_test = sc.AnnData(test_M)

        sc.pp.log1p(adata_train)
        sc.pp.log1p(adata_test)

        adata_merged = anndata.concat(
            [adata_train, adata_test],
            join='outer',
            label='dataset',
            keys=['train', 'test']
        )
        sc.pp.highly_variable_genes(adata_merged, n_top_genes=cfg.hvg, subset=True, batch_key='dataset')
        adata_train = adata_merged[adata_merged.obs['dataset'] == 'train'].copy()
        adata_test = adata_merged[adata_merged.obs['dataset'] == 'test'].copy()
        train_df = pd.DataFrame(adata_train.X, index=adata_train.obs_names, columns=adata_train.var_names)
        test_df = pd.DataFrame(adata_test.X, index=adata_test.obs_names, columns=adata_test.var_names)
    else:
        train_df = train_M
        test_df = test_M

    return train_df, train_L, train_df, test_df, test_L, test_df

import sys

from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import os

pandas2ri.activate()


def get_mat_path(sc_path, cfg, output_path_list):

    output_path_meta = output_path_list[0]
    output_path_cp = output_path_list[1]
    output_path_eip = output_path_list[2]
    output_path_os = output_path_list[3]
    output_path_di = output_path_list[4]
    output_path_dr = output_path_list[5]

    importr("GSEABase")
    importr("AUCell")
    importr("SingleCellExperiment")
    print('R package load!')
    r.source('./AucForPy.R')

    cfg.refresh = 1

    output_paths = [output_path_meta, output_path_cp, output_path_eip, output_path_os, output_path_di, output_path_dr]
    r.AUC(sc_path, 'Reactome_Human+PharmGKB', output_paths, cfg.refresh)

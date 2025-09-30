import sys, importlib
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
from glob import glob
import scipy
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score, roc_auc_score, get_scorer_names
from sklearn.preprocessing import LabelEncoder


class ntuple:
    def __init__(self, filename, pot = 0, trigger = 0):
        self.filename = filename
        self.pot = pot
        self.trigger =  trigger
        data = uproot.open(self.filename)
        f_pot = data["summed_pot"].member("fVal")
        self.is_mc = True
        if f_pot == 0.0:
            self.is_mc = False
        else:
            self.pot = f_pot
        print(f"pot = {self.pot}; trigger = {self.trigger}")
        self.branch_name = [
            "trk_distance_v",
            "trk_len_v",
            "trk_score_v",
            "trk_llr_pid_score_v",
            "trk_pid_chipr_v",
            "trk_energy_proton_v",
            "trk_range_muon_mom_v",
            "trk_mcs_muon_mom_v",
            "pfnhits",
            "pfp_generation_v",
            "MC_Signal",
            "nslice",
            "CosmicIP",
            "topological_score",
            "reco_nu_vtx_sce_x",
            "reco_nu_vtx_sce_y",
            "reco_nu_vtx_sce_z",
            "trk_score_v",
            "pfp_generation_v",
            "trk_sce_start_x_v",
            "trk_sce_start_y_v",
            "trk_sce_start_z_v",
            "trk_sce_end_x_v",
            "trk_sce_end_y_v",
            "trk_sce_end_z_v",
            "pfp_trk_daughters_v",
            "pfp_shr_daughters_v",
            "backtracked_pdg",
            "mc_pdg"
        ]
        # track level branches
        self.branch_name_trk = [
            "trk_distance_v",
            "trk_len_v",
            "trk_score_v",
            "trk_llr_pid_score_v",
            "trk_pid_chipr_v",
            "trk_energy_proton_v",
            "trk_range_muon_mom_v",
            "trk_mcs_muon_mom_v",
            "pfnhits",
            "pfp_generation_v",
            "trk_score_v",
            "pfp_generation_v",
            "trk_sce_start_x_v",
            "trk_sce_start_y_v",
            "trk_sce_start_z_v",
            "trk_sce_end_x_v",
            "trk_sce_end_y_v",
            "trk_sce_end_z_v",
            "pfp_trk_daughters_v",
            "pfp_shr_daughters_v"
        ]
        self.branch_name_trk_mc_only = [
            "backtracked_pdg"
        ]
        # event level branches
        self.branch_name_evt = [
            "MC_Signal",
            "nslice",
            "CosmicIP",
            "topological_score",
            "reco_nu_vtx_sce_x",
            "reco_nu_vtx_sce_y",
            "reco_nu_vtx_sce_z",
        ]
        self.weight_name_evt = [
            "spline_weight",
            "tuned_cv_weight"
        ]
        tree = data["stv_tree"]
        self.branches = tree.arrays(self.branch_name, library="ak")
        self.branches_trk = tree.arrays(self.branch_name_trk, library="ak")
        self.branches_trk_mc = tree.arrays(self.branch_name_trk_mc_only, library="ak")
        self.branches_evt = tree.arrays(self.branch_name_evt, library="ak")
        self.weight_evt = tree.arrays(self.weight_name_evt, library="ak")
    def _apply_cut_trk(self, mask):
        len_before1 = len(ak.flatten(self.branches_trk["trk_distance_v"]))
        len_before2 = len(ak.flatten(mask))
        self.branches_trk = self.branches_trk[mask]
        if len(ak.flatten(self.branches_trk_mc["backtracked_pdg"])) !=0:
            self.branches_trk_mc = self.branches_trk_mc[mask]
        len_after1 = len(ak.flatten(self.branches_trk["trk_distance_v"]))
        len_after2 = len(ak.flatten(mask))
        print(f"_apply_cut_trk: length before: {len_before1} {len_before2}")
        print(f"_apply_cut_trk: length after: {len_after1} {len_after2}")
        self._map_weight_to_trk()
    def _apply_cut_evt(self, mask):
        len_before1 = len(mask)
        len_before2 = len(self.branches_evt["nslice"])
        self.branches_evt = self.branches_evt[mask]
        self.branches_trk = self.branches_trk[mask]
        self.branches_trk_mc = self.branches_trk_mc[mask]
        self.branches = self.branches[mask]
        self.weight_evt = self.weight_evt[mask]
        len_after1 = len(mask)
        len_after2 = len(self.branches_evt["nslice"])
        print(f"_apply_cut_evt: length before: {len_before1} {len_before2}")
        print(f"_apply_cut_evt: length after: {len_after1} {len_after2}")

    def _map_weight_to_trk(self):
        weight_trk = []
        for w in self.weight_name_evt:
            w_val = self.weight_evt[w] 
            trk_temp = self.branches_trk["trk_len_v"]
            w_trk = [[w_val[i]] * len(sublist) for i, sublist in enumerate(trk_temp)]
            weight_trk.append(w_trk)
        self.weight_trk = ak.zip({self.weight_name_evt[0]: weight_trk[0], self.weight_name_evt[1]: weight_trk[1]})

    def _add_new_branch(self):
        # add new branch, pfp_num_daughter and range_mcs_difference
        pfp_num_daughter = self.branches_trk["pfp_trk_daughters_v"] + self.branches_trk["pfp_shr_daughters_v"]
        range_mcs_difference = (self.branches_trk["trk_range_muon_mom_v"] - self.branches_trk["trk_mcs_muon_mom_v"])/self.branches_trk["trk_range_muon_mom_v"]
        self.branches_trk = ak.with_field(self.branches_trk, pfp_num_daughter, "pfp_num_daughter")
        self.branches_trk = ak.with_field(self.branches_trk, range_mcs_difference, "range_mcs_difference")


    def get_trk_feature(self, name):
        weight = self.weight_trk[self.weight_name_evt[0]] * self.weight_trk[self.weight_name_evt[1]]
        return [ak.flatten(self.branches_trk[name]), ak.flatten(weight)]

    def get_trk_feature_pdg(self, name):
        if self.is_mc:
            flatten_bt_pdg = ak.flatten(self.branches_trk_mc["backtracked_pdg"])
            mask_muon = (np.abs(flatten_bt_pdg) == 13)
            mask_prot = (np.abs(flatten_bt_pdg) == 2212)
            mask_pion = (np.abs(flatten_bt_pdg) == 211)
            mask_other = (np.abs(flatten_bt_pdg) != 13) & (np.abs(flatten_bt_pdg) != 2212) & (np.abs(flatten_bt_pdg) != 211)
            print(len(mask_muon), len(mask_prot), len(mask_pion), len(mask_other))
            feature = ak.flatten(self.branches_trk[name])
            weight = ak.flatten(self.weight_trk[self.weight_name_evt[0]] * self.weight_trk[self.weight_name_evt[1]])
            output_feature = [feature[mask_other], feature[mask_pion], feature[mask_prot], feature[mask_muon]]
            output_weight  = [weight[mask_other], weight[mask_pion], weight[mask_prot], weight[mask_muon]]
            #return [feature[mask_other], feature[mask_pion], feature[mask_prot], feature[mask_muon], weight[mask_other], weight[mask_pion], weight[mask_prot], weight[mask_muon]]
            return [output_feature, output_weight]


    def plot1d(self, bname, bins=100, xrange=(-1,1), title = ""):
        x_title = bname
        y_title = "Events"
        x = ak.flatten(self.branches_trk[bname])
        plt.figure(figsize=(7.0, 4.5), dpi = 200)
        plt.hist(x, bins=bins, range=xrange)
        plt.xlabel(rf"{x_title}")
        plt.ylabel(rf"{y_title}")
        plt.show()




import numpy as np
import uproot
import awkward as ak
import pandas as pd
import matplotlib.pyplot as plt
import json
import os, sys

class ntuple:
    def __init__(self, filename, pot=0, trigger = 0):
        self.filename = filename
        self.pot = pot
        self.trigger =  trigger
        print(f"Loading ntuple {self.filename}")

    def _load_branch(self):
        # event level branch
        # 1. reco
        # 2. true

        # track/shower branch
        # 1. reco
        # 2. true
        data = uproot.open(self.filename)
        f_pot = data["summed_pot"].member("fVal")
        self.is_mc = False
        if f_pot != 0:
            self.pot = f_pot
            self.is_mc = True
        
        tree = data["stv_tree"]
        print(f"  POT = {self.pot}; Trigger = {self.trigger}")
        script_path = os.path.dirname(os.path.abspath(__file__))
        self.branches = json.load(open(os.path.dirname(script_path) + "/config/branches.json"))
        self.branch_reco_evt = tree.arrays(self.branches["reco"]["event"], library="ak")
        self.branch_reco_trk = tree.arrays(self.branches["reco"]["track"], library="ak")
        self.branch_true_evt = tree.arrays(self.branches["true"]["event"], library="ak")
        self.branch_true_trk = tree.arrays(self.branches["true"]["track"], library="ak")
        self.branch_weight_evt   = tree.arrays(self.branches["weight"], library="ak")
        try:
            self.branch_true_signal = tree.arrays(self.branches["true"]["signal"], library="ak")
        except uproot.KeyInFileError as e:
            print(f"Skip true signal branches for non-overlay!")
        self._broadcast_weight_to_trk()


    def save_to_csv(self, filepath):
        # save branches to csv
        df = ak.to_dataframe(self.branch_reco_evt)
        df.to_csv(filepath + "/reco_evt.csv")
        df = ak.to_dataframe(self.branch_reco_trk)
        df.to_csv(filepath + "/reco_trk.csv")
        df = ak.to_dataframe(self.branch_true_evt)
        df.to_csv(filepath + "/true_evt.csv")
        df = ak.to_dataframe(self.branch_true_trk)
        df.to_csv(filepath + "/true_trk.csv")
        df = ak.to_dataframe(self.branch_weight_evt)
        df.to_csv(filepath + "/weight_evt.csv")
        df = ak.to_dataframe(self.branch_weight_trk)
        df.to_csv(filepath + "/weight_trk.csv")

    def _broadcast_weight_to_trk(self):
        weight_trk = []
        trk_temp = self.branch_reco_trk[self.branches["reco"]["track"][0]]  # using the first branch as template
        for w in self.branches["weight"]:
            w_val = self.branch_weight_evt[w]
            # broadcast event level weights to track level weights, the tracks in each event have the same weights
            w_trk = ak.broadcast_arrays(trk_temp, w_val)[1]
            weight_trk.append(w_trk)
        self.branch_weight_trk = ak.zip({self.branches["weight"][0]: weight_trk[0], self.branches["weight"][1]: weight_trk[1]})

    def _apply_cut_trk(self, mask):

        len_before1 = len(ak.flatten(self.branch_reco_trk["trk_distance_v"]))
        len_before2 = len(ak.flatten(mask))

        # apply the cut
        self.branch_reco_trk = self.branch_reco_trk[mask]
        self.branch_weight_trk = self.branch_weight_trk[mask]
        if len( ak.flatten( self.branch_true_trk['backtracked_pdg'] ) ):
            self.branch_true_trk = self.branch_true_trk[mask]

        len_after1 = len(ak.flatten(self.branch_reco_trk["trk_distance_v"]))
        len_after2 = len(ak.flatten(mask))
        print(f"_apply_cut_trk: length before: {len_before1} {len_before2}")
        print(f"_apply_cut_trk: length after: {len_after1} {len_after2}")

    def _apply_cut_evt(self, mask):
        len_before1 = len(mask)
        len_before2 = len(self.branch_reco_evt[self.branches["reco"]["event"][0]])
        # apply the cut
        self.branch_reco_evt = self.branch_reco_evt[mask]
        self.branch_reco_trk = self.branch_reco_trk[mask]
        self.branch_weight_evt = self.branch_weight_evt[mask]
        self.branch_weight_trk = self.branch_weight_trk[mask]

        self.branch_true_evt = self.branch_true_evt[mask]
        self.branch_true_trk = self.branch_true_trk[mask]

        len_after1 = len(mask)
        len_after2 = len(self.branch_reco_evt[self.branches["reco"]["event"][0]])
        print(f"_apply_cut_evt: length before: {len_before1} {len_before2}")
        print(f"_apply_cut_evt: length after: {len_after1} {len_after2}")

    def get_trk_feature_pdg(self, name):
        if self.is_mc:
            flatten_bt_pdg = ak.flatten(self.branch_true_trk["backtracked_pdg"])
            mask_muon = (np.abs(flatten_bt_pdg) == 13)
            mask_prot = (np.abs(flatten_bt_pdg) == 2212)
            mask_pion = (np.abs(flatten_bt_pdg) == 211)
            mask_other = (np.abs(flatten_bt_pdg) != 13) & (np.abs(flatten_bt_pdg) != 2212) & (np.abs(flatten_bt_pdg) != 211)

            print(len(mask_muon), len(mask_prot), len(mask_pion), len(mask_other))
            feature = ak.flatten(self.branch_reco_trk[name])
            weight = ak.flatten(self.branch_weight_trk[self.branches["weight"][0]] * self.branch_weight_trk[self.branches["weight"][1]])
            output_feature = [feature[mask_other], feature[mask_pion], feature[mask_prot], feature[mask_muon]]
            output_weight  = [weight[mask_other], weight[mask_pion], weight[mask_prot], weight[mask_muon]]
            #return [feature[mask_other], feature[mask_pion], feature[mask_prot], feature[mask_muon], weight[mask_other], weight[mask_pion], weight[mask_prot], weight[mask_muon]]
            return [output_feature, output_weight]

    def get_trk_feature(self, name):
        weight = self.branch_weight_trk[self.branches["weight"][0]] * self.branch_weight_trk[self.branches["weight"][1]]
        return [ak.flatten(self.branch_reco_trk[name]), ak.flatten(weight)]

    def add_true_evt_branch(self, name, arr):
        self.branches["true"]["event"].append(name)
        self.branch_true_evt = ak.with_field(self.branch_true_evt, arr, name)
    def add_true_trk_branch(self, name, arr):
        self.branches["true"]["track"].append(name)
        self.branch_true_trk = ak.with_field(self.branch_true_trk, arr, name)

    def add_reco_evt_branch(self, name, arr):
        self.branches["reco"]["event"].append(name)
        self.branch_reco_evt = ak.with_field(self.branch_reco_evt, arr, name)
    def add_reco_trk_branch(self, name, arr):
        self.branches["reco"]["track"].append(name)
        self.branch_reco_trk = ak.with_field(self.branch_reco_trk, arr, name)

    def plot1d(self, bname, bins=50, xrange=None, title = ""):
        x_title = bname
        y_title = "Events"
        merged = ak.zip({**{f: self.branch_reco_evt[f] for f in self.branch_reco_evt.fields},
                         **{f: self.branch_reco_trk[f] for f in self.branch_reco_trk.fields}})
        x = ak.flatten(merged[bname])
        if xrange == None:
            xrange=(min(x), max(x))
            print(min(x), max(x))
        plt.figure(figsize=(7.0, 4.5), dpi = 200)
        plt.hist(x, bins=bins, range=xrange)
        plt.xlabel(rf"{x_title}")
        plt.ylabel(rf"{y_title}")
        plt.show()

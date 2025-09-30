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
import numpy as np
import matplotlib.pyplot as plt
import json, os
from collections import defaultdict

from . import selection
from . import ntuple
from . import bdt

class ccxp:
    def __init__(self, run = ["run1"]):
        self.run = run
        self.runs = ""
        for r in self.run:
            self.runs = self.runs + "_" + r
        script_path = os.path.dirname(os.path.abspath(__file__))
        f = open(script_path + "/config/ntuple.json")
        self.data = json.load(f)
        self.samples = {}
        for r in self.run:
            self.samples[r] = ["beamon", "beamoff", "overlay", "dirt"]
        self._load_data()

    def _load_data(self):
        self.ntuple = defaultdict(lambda: defaultdict(selection.selection))
        for run, ss in self.samples.items():
            for s in ss:
                if s == "beamon":
                    self.ntuple[run][s] = selection.selection(self.data["path"] + self.data[run][s]["name"], pot = self.data[run][s]["pot"], trigger=self.data[run][s]["trigger"])
                elif s == "beamoff":
                    self.ntuple[run][s] = selection.selection(self.data["path"] + self.data[run][s]["name"], trigger=self.data[run][s]["trigger"])
                else:
                    self.ntuple[run][s] = selection.selection(self.data["path"] + self.data[run][s]["name"])
                self.ntuple[run][s].apply_cut()
                self.ntuple[run][s].data._add_new_branch()

    def get_bdt_feature(self, feature_branch = ["trk_distance_v"]):
        bdt_feature = {}
        for r in self.run:
            for b in feature_branch:
                arr = ak.flatten(self.ntuple[r]["overlay"].data.branches_trk[b])
                if b in bdt_feature:
                    bdt_feature[b] = ak.concatenate([bdt_feature[b], arr])
         
                else:
                    bdt_feature[b] = arr

        backtracked_pdg = ak.Array([])
        for r in self.run:
            b = "backtracked_pdg"
            arr = ak.flatten(self.ntuple[r]["overlay"].data.branches_trk_mc[b])
            backtracked_pdg = ak.concatenate([backtracked_pdg, arr])

        for b, arr in bdt_feature.items():
            bdt_feature[b] = np.clip(np.array(arr), -1e16, 1e16)
        bdt_feature["backtracked_pdg"] = np.clip(np.array(backtracked_pdg), -1e16, 1e16)

        return bdt_feature

    def plot(self, branch_name, title="title;x;y", bins=50, xrange=(-1, 1), save=False):
        branch_val = defaultdict(lambda: defaultdict())
        scale_factor = defaultdict(lambda: defaultdict())
        val_plot = {}
        weight_plot = {}
        for run, ss in self.samples.items():
            for s in ss:
                if s == "overlay":
                    branch_val[run][s] = self.ntuple[run][s].data.get_trk_feature_pdg(branch_name)
                else:
                    branch_val[run][s] = self.ntuple[run][s].data.get_trk_feature(branch_name)
        
                if s == "beamoff":
                    scale_factor[run][s] = self.ntuple[run]["beamon"].data.trigger/self.ntuple[run][s].data.trigger
                else:
                    scale_factor[run][s] = self.ntuple[run]["beamon"].data.pot/self.ntuple[run][s].data.pot
        
                if s in val_plot:
                    if s == "overlay":
                        for i in range(len(val_plot[s])):
                            val_plot[s][i] =  ak.concatenate([val_plot[s][i], branch_val[run][s][0][i]])
                    else:
                        val_plot[s] = ak.concatenate([val_plot[s], branch_val[run][s][0]])
                else:
                    val_plot[s] = branch_val[run][s][0]
        
                if s in weight_plot:
                    if s == "overlay":
                        for i in range(len(weight_plot[s])):
                            weight_plot[s][i] =  ak.concatenate([weight_plot[s][i], branch_val[run][s][1][i]*scale_factor[run][s]])
                    else:
                        weight_plot[s] = ak.concatenate([weight_plot[s], branch_val[run][s][1]*scale_factor[run][s]])
                else:                    
                    weight_plot[s] = branch_val[run][s][1]
                    if s == "overlay":
                        for i in range(len(weight_plot[s])):
                            weight_plot[s][i] = weight_plot[s][i]*scale_factor[run][s]
                    else:
                        weight_plot[s] = weight_plot[s]*scale_factor[run][s]
        
        overlay_val_plot=[val_plot["beamoff"], val_plot["dirt"], val_plot["overlay"][0], val_plot["overlay"][1], val_plot["overlay"][2], val_plot["overlay"][3]]
        overlay_weight_plot=[weight_plot["beamoff"], weight_plot["dirt"], weight_plot["overlay"][0], weight_plot["overlay"][1], weight_plot["overlay"][2], weight_plot["overlay"][3]]
        
        overlay_weight_plot_fixed = [ak.where(np.isinf(w), 1, w) for w in overlay_weight_plot]

        # Compute histogram (not drawn yet)
        counts, bins = np.histogram(val_plot["beamon"], bins=bins, range=xrange)

        # Bin centers
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        # Poisson errors for counts
        errors = np.sqrt(counts)
        plt.figure(figsize=(7.0, 4.5), dpi = 200)
        # Plot as points with error bars
        plt.errorbar(bin_centers, counts, yerr=errors, fmt='o', color='black', capsize=3, label="Data")

        # (Optional) overlay step-style histogram for reference
        #plt.step(bins[:-1], counts, where='mid', color='blue', alpha=0.5, label="Histogram")
        labels = ["BeamOff", "Dirt", "Other", "pion", "proton", "muon"]
        
        plt.hist(
            overlay_val_plot,
            bins=bins,
            stacked=True,
            alpha=0.8,
            weights=overlay_weight_plot_fixed,
            label=labels,
            range=xrange
        )
        pot= 0
        for run, ss in self.samples.items():
            pot = pot + self.ntuple[run]["beamon"].data.pot
        plt.title(f"POT = {pot}")
        plt.xlabel(branch_name)
        plt.ylabel("Entries per bin")
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(f"/exp/uboone/app/users/liangliu/analysis-code/tutorial/script/ccxp/plot/beamon_overlay{self.runs}_{branch_name}.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"/exp/uboone/app/users/liangliu/analysis-code/tutorial/script/ccxp/plot/beamon_overlay{self.runs}_{branch_name}.pdf", dpi=300, bbox_inches='tight')
        plt.show()



    def selection(self):
        bdt_t = bdt.bdt()
        predict_pdg = defaultdict(lambda: defaultdict())
        for r, ss in self.samples.items():
            for s in ss:
                mask = self.ntuple[r][s].apply_bdt_selection()
                self.ntuple[r][s].data._apply_cut_evt(mask)
                # get bdt prediction
                pred_pdg = bdt_t.predict(self.ntuple[r][s])
                # remove track not proton and muon
                mask = ((pred_pdg == 1) | (pred_pdg == 3))
                predict_pdg[r][s] = pred_pdg[mask]
                self.ntuple[r][s].data._apply_cut_trk(mask)
                # remove events that have more than on muon
                mask=predict_pdg[r][s][predict_pdg[r][s] == 1]
                mask=ak.num(mask) == 1
                self.ntuple[r][s].data._apply_cut_evt(mask)
                predict_pdg[r][s] = predict_pdg[r][s][mask]
                self.ntuple[r][s].data._map_weight_to_trk()
                # apply topology score cut
                topo_score = self.ntuple[r][s].data.branches_evt["topological_score"]
                proton_multi = ak.num(predict_pdg[r][s][predict_pdg[r][s] == 3])
                mask = ((proton_multi == 0) & (topo_score > 0.2)) | ((proton_multi == 1) & (topo_score > 0.2)) | (proton_multi > 1)
                self.ntuple[r][s].data._apply_cut_evt(mask)
                self.ntuple[r][s].data._map_weight_to_trk()


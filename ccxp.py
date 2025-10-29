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


from .src.ntuple import ntuple
from .src.selection import selection
from .utils.plotter import track_plotter


class ccxp:
    def __init__(self):
        self.script_path = os.path.dirname(os.path.abspath(__file__))
        f = open(self.script_path + "/config/ntuple.json")
        self.files = json.load(f)

    def processing_ntuple(self, run = ["run1"]):
        self.run = run
        self.runs = ""
        self.ntuples = defaultdict(lambda: defaultdict(ntuple))
        for r in self.run:
            self.runs = self.runs + "_" + r
            for s, val in self.files[r].items():
                if "detVar" in s: continue
                if s not in ["overlay", "beamoff", "beamon", "dirt"]: continue
                if s == 'beamon': 
                    self.ntuples[r][s] = ntuple(self.files['path'] + self.files[r][s]['name'], pot = self.files[r][s]['pot'], trigger = self.files[r][s]['trigger'])
                elif s == "beamoff":
                    self.ntuples[r][s] = ntuple(self.files['path'] + self.files[r][s]['name'], trigger = self.files[r][s]['trigger'])
                else:
                    self.ntuples[r][s] = ntuple(self.files['path'] + self.files[r][s]['name'])

                self.ntuples[r][s]._load_branch()
                sel = selection(self.ntuples[r][s])
                sel.execute()
                #outpath = self.script_path + "/data/" + r + "/" +  s
                #print(outpath) 

    def plot_track(self, branch_name, title="title;x;y", bins=50, xrange=(-1, 1), save=False):
        data, mc, mc_weight = self._construct_track_feature(branch_name)
        pot = 0
        for r in self.run:
            pot += self.ntuples[r]["beamon"].pot
        track_plotter(branch_name, data, mc, mc_weight, pot=pot, title=title, bins=bins, xrange=xrange, save=save)

    def _construct_track_feature(self, branch_name):

        branch_val = defaultdict(lambda: defaultdict())
        scale_factor = defaultdict(lambda: defaultdict())
        val_plot = {}
        weight_plot = {}

        for r in self.run:
            for s, val in self.files[r].items():
                if s not in ["overlay", "beamoff", "beamon", "dirt"]: continue
                print(s)
                if s == "overlay":
                    branch_val[r][s] = self.ntuples[r][s].get_trk_feature_pdg(branch_name)
                else:
                    branch_val[r][s] = self.ntuples[r][s].get_trk_feature(branch_name)
        
                if s == "beamoff":
                    scale_factor[r][s] = self.ntuples[r]["beamon"].trigger/self.ntuples[r][s].trigger
                else:
                    scale_factor[r][s] = self.ntuples[r]["beamon"].pot/self.ntuples[r][s].pot
        
                if s in val_plot:
                    if s == "overlay":
                        for i in range(len(val_plot[s])):
                            val_plot[s][i] =  ak.concatenate([val_plot[s][i], branch_val[r][s][0][i]])
                    else:
                        val_plot[s] = ak.concatenate([val_plot[s], branch_val[r][s][0]])
                else:
                    val_plot[s] = branch_val[r][s][0]
        
                if s in weight_plot:
                    if s == "overlay":
                        for i in range(len(weight_plot[s])):
                            weight_plot[s][i] =  ak.concatenate([weight_plot[s][i], branch_val[r][s][1][i]*scale_factor[r][s]])
                    else:
                        weight_plot[s] = ak.concatenate([weight_plot[s], branch_val[r][s][1]*scale_factor[r][s]])
                else:                    
                    weight_plot[s] = branch_val[r][s][1]
                    if s == "overlay":
                        for i in range(len(weight_plot[s])):
                            weight_plot[s][i] = weight_plot[s][i]*scale_factor[r][s]
                    else:
                        weight_plot[s] = weight_plot[s]*scale_factor[r][s]
        
        overlay_val_plot=[val_plot["beamoff"], val_plot["dirt"], val_plot["overlay"][0], val_plot["overlay"][1], val_plot["overlay"][2], val_plot["overlay"][3]]
        overlay_weight_plot=[weight_plot["beamoff"], weight_plot["dirt"], weight_plot["overlay"][0], weight_plot["overlay"][1], weight_plot["overlay"][2], weight_plot["overlay"][3]]
        overlay_weight_plot_fixed = [ak.where(np.isinf(w), 1, w) for w in overlay_weight_plot]

        return val_plot["beamon"], overlay_val_plot, overlay_weight_plot_fixed




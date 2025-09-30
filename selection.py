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
from . import ntuple

class selection:
    def __init__(self, filename, pot = 0, trigger = 0):
        self.filename = filename
        self.nslice_cut = 1
        self.cosmic_ip_cut = 10

        self.FV_x_min = 21.50
        self.FV_x_max = 234.85
        self.FV_y_min = -95.00
        self.FV_y_max = 95.00
        self.FV_z_min = 21.50
        self.FV_z_max = 966.80

        self.CV_x_min = 10
        self.CV_x_max = 246.35
        self.CV_y_min = -106.5
        self.CV_y_max = 106.5
        self.CV_z_min = 10
        self.CV_z_max = 1026.8

        self.trk_score_cut = 0.5
        self.topological_score_cut = 0.1
        self.data = ntuple.ntuple(self.filename, pot=pot, trigger = trigger)
    def apply_cut(self):
        self.data._apply_cut_evt(self.pre_selection())
        self.data._apply_cut_trk(self.filter_containment_trk())

    def _inside_fv(self, x, y, z):
        return (x > self.FV_x_min) & (x < self.FV_x_max) & (y > self.FV_y_min) & (y < self.FV_y_max) & (z > self.FV_z_min) & (z < self.FV_z_max)
    def _inside_cv(self, x, y, z):
        return (x > self.CV_x_min) & (x < self.CV_x_max) & (y > self.CV_y_min) & (y < self.CV_y_max) & (z > self.CV_z_min) & (z < self.CV_z_max)

    def pre_selection(self):
        nslice = self.data.branches["nslice"]
        cosmic_ip = self.data.branches["CosmicIP"]
        topological_score = self.data.branches["topological_score"]

        reco_nu_vtx_sce_x = self.data.branches["reco_nu_vtx_sce_x"]
        reco_nu_vtx_sce_y = self.data.branches["reco_nu_vtx_sce_y"]
        reco_nu_vtx_sce_z = self.data.branches["reco_nu_vtx_sce_z"]
        nu_in_fv = self._inside_fv(reco_nu_vtx_sce_x, reco_nu_vtx_sce_y, reco_nu_vtx_sce_z)

        trk_score_v = self.data.branches["trk_score_v"]
        pfp_generation_v = self.data.branches["pfp_generation_v"]
        trk_score_v = trk_score_v[pfp_generation_v == 2].to_list()
        ncols = [min(row) if row else 0 for row in trk_score_v]
        mask = (nslice == self.nslice_cut) & (cosmic_ip > self.cosmic_ip_cut) & nu_in_fv & (~(np.array(ncols) < self.trk_score_cut))
        return mask

    def filter_containment_trk(self):
        pfp_generation_v = self.data.branches_trk["pfp_generation_v"]
        trk_sce_start_x_v = self.data.branches_trk["trk_sce_start_x_v"]
        trk_sce_start_y_v = self.data.branches_trk["trk_sce_start_y_v"]
        trk_sce_start_z_v = self.data.branches_trk["trk_sce_start_z_v"]
        trk_sce_end_x_v = self.data.branches_trk["trk_sce_end_x_v"]
        trk_sce_end_y_v = self.data.branches_trk["trk_sce_end_y_v"]
        trk_sce_end_z_v = self.data.branches_trk["trk_sce_end_z_v"]
        mask1 = self._inside_cv(trk_sce_start_x_v, trk_sce_start_y_v, trk_sce_start_z_v)
        mask2 = self._inside_cv(trk_sce_end_x_v, trk_sce_end_y_v, trk_sce_end_z_v)
        return (pfp_generation_v == 2) & mask1 & mask2


    ## apply the DBT selection
    def apply_bdt_selection(self):
        # veto the non prime track
        trk_sce_start_x_v = self.data.branches_trk["trk_sce_start_x_v"]
        trk_sce_start_y_v = self.data.branches_trk["trk_sce_start_y_v"]
        trk_sce_start_z_v = self.data.branches_trk["trk_sce_start_z_v"]
        trk_sce_end_x_v = self.data.branches_trk["trk_sce_end_x_v"]
        trk_sce_end_y_v = self.data.branches_trk["trk_sce_end_y_v"]
        trk_sce_end_z_v = self.data.branches_trk["trk_sce_end_z_v"]
        mask1 = self._inside_cv(trk_sce_start_x_v, trk_sce_start_y_v, trk_sce_start_z_v)
        mask2 = self._inside_cv(trk_sce_end_x_v, trk_sce_end_y_v, trk_sce_end_z_v)
        mask = mask1 & mask2
        mask = ak.num(mask) > 0 & ak.all(mask, axis=1) # there must be at least one track and all tracks must be containment.
        return mask







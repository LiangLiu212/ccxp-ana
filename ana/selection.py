#from . import ntuple
import awkward as ak
from .BDT import bdt

class selection:
    def __init__(self, nt):
        
        self.ntuple = nt
        
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
    def execute(self):
        self.pre_selection()
        self.filter_containment_trk()
        self.apply_bdt()

    def _inside_fv(self, x, y, z):
        return (x > self.FV_x_min) & (x < self.FV_x_max) & (y > self.FV_y_min) & (y < self.FV_y_max) & (z > self.FV_z_min) & (z < self.FV_z_max)
    def _inside_cv(self, x, y, z):
        return (x > self.CV_x_min) & (x < self.CV_x_max) & (y > self.CV_y_min) & (y < self.CV_y_max) & (z > self.CV_z_min) & (z < self.CV_z_max)

    def pre_selection(self):

        # nslice
        mask = self.ntuple.branch_reco_evt["nslice"] == self.nslice_cut
        self.ntuple._apply_cut_evt(mask)

        pfp_generation_v = self.ntuple.branch_reco_trk["pfp_generation_v"]
        mask = pfp_generation_v == 2
        self.ntuple._apply_cut_trk(mask)

        cosmic_ip = self.ntuple.branch_reco_evt["CosmicIP"]
        topological_score = self.ntuple.branch_reco_evt["topological_score"]
        
        reco_nu_vtx_sce_x = self.ntuple.branch_reco_evt["reco_nu_vtx_sce_x"]
        reco_nu_vtx_sce_y = self.ntuple.branch_reco_evt["reco_nu_vtx_sce_y"]
        reco_nu_vtx_sce_z = self.ntuple.branch_reco_evt["reco_nu_vtx_sce_z"]
        
        nu_in_fv = self._inside_fv(reco_nu_vtx_sce_x, reco_nu_vtx_sce_y, reco_nu_vtx_sce_z)
        
        trk_score_v = self.ntuple.branch_reco_trk["trk_score_v"]
        trk_like = ak.min(trk_score_v, axis=1) > self.trk_score_cut

        mask = (cosmic_ip > self.cosmic_ip_cut) & trk_like

        self.ntuple._apply_cut_evt(mask)

    def filter_containment_trk(self):
        trk_sce_start_x_v = self.ntuple.branch_reco_trk["trk_sce_start_x_v"]
        trk_sce_start_y_v = self.ntuple.branch_reco_trk["trk_sce_start_y_v"]
        trk_sce_start_z_v = self.ntuple.branch_reco_trk["trk_sce_start_z_v"]
        trk_sce_end_x_v = self.ntuple.branch_reco_trk["trk_sce_end_x_v"]
        trk_sce_end_y_v = self.ntuple.branch_reco_trk["trk_sce_end_y_v"]
        trk_sce_end_z_v = self.ntuple.branch_reco_trk["trk_sce_end_z_v"]
        mask1 = self._inside_cv(trk_sce_start_x_v, trk_sce_start_y_v, trk_sce_start_z_v)
        mask2 = self._inside_cv(trk_sce_end_x_v, trk_sce_end_y_v, trk_sce_end_z_v)
        mask = mask1 & mask2
        self.ntuple._apply_cut_trk(mask)

    def apply_bdt(self):
        model = bdt()
        print("Hello BDT!")

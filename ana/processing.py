import awkward as ak
import vector
import numpy as np

def define_signal(r, s, nt):
    if s != "overlay":
        return
    print(r, s, nt)
    pdg = nt.branch_true_signal["mc_pdg"]
    # only cc interaction
    
    mask = (nt.branch_true_evt["mc_ccnc"] == 0)
    # only proton, neutron, muon, and charged-pion in the final states
    # pdg_mask = (np.abs(pdg) != 2212) & (np.abs(pdg) != 2112) & (np.abs(pdg) != 13) & (np.abs(pdg) != 211)
    pdg_mask = (np.abs(pdg) != 2212) & (np.abs(pdg) != 2112) & (np.abs(pdg) != 13)
    mask = mask & (ak.num(pdg[pdg_mask]) == 0)
    # only one muon in the final states
    mask = mask & (ak.num(pdg[pdg == 13]) == 1)
    
    # construct the lorentz vector with python-vector and awkward
    vector.register_awkward()
    px = nt.branch_true_signal["mc_px"]
    py = nt.branch_true_signal["mc_py"]
    pz = nt.branch_true_signal["mc_pz"]
    e = nt.branch_true_signal["mc_E"]
    true_vec = ak.zip(
        {
            "px": px,
            "py": py,
            "pz": pz,
            "e": e,
        },
        with_name="Momentum4D"
    )
    
    # the momentum of muon should be within [0.1, 1.2] GeV
    muon_rho = ak.fill_none(ak.max(true_vec[np.abs(pdg) == 13].rho, axis=1), 0)
    mask = mask & (muon_rho > 0.1) & (muon_rho < 1.2)
    
    # the momentum of most energitic pion should be less than 
    pion_rho = ak.fill_none(ak.max(true_vec[np.abs(pdg) == 211].rho, axis=1), 0)
    mask = mask & (pion_rho < 0.07)
    nt.add_true_evt_branch("is_mc_signal", mask)



def categorize_event(r, s, nt):
    if s != "overlay": return
    print(nt.branches)


def construct_new_track_feature(r, s, nt):
    pfp_num_daughter = nt.branch_reco_trk['pfp_trk_daughters_v'] + nt.branch_reco_trk['pfp_shr_daughters_v']
    range_mcs_difference = (nt.branch_reco_trk['trk_range_muon_mom_v'] - nt.branch_reco_trk['trk_mcs_muon_mom_v'])/nt.branch_reco_trk['trk_range_muon_mom_v'] 
    nt.add_reco_trk_branch("pfp_num_daughter", pfp_num_daughter)
    nt.add_reco_trk_branch("range_mcs_difference", range_mcs_difference)

def get_reco_event_feature(r, s, nt, name):
    if s == "overlay":
        feature = nt.branch_reco_evt[name]
        weight = nt.branch_weight_evt[nt.branches["weight"][0]] * nt.branch_weight_evt[nt.branches["weight"][1]]
        mask_signal = nt.branch_true_evt["is_mc_signal"]
        output_feature = [feature[~mask_signal], feature[mask_signal]]
        output_weight = [weight[~mask_signal], weight[mask_signal]]
        return [output_feature, output_weight]
    else:
        weight = nt.branch_weight_evt[nt.branches["weight"][0]] * nt.branch_weight_evt[nt.branches["weight"][1]]
        return [nt.branch_reco_evt[name], weight]



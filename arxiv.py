    def plot(self, branch_name, title="title;x;y", bins=50, xrange=(-1, 1)):

        branch_val = {}
        for s in self.samples:
            if s == "overlay":
                branch_val[s] = self.ntuple[s].data.get_trk_feature_pdg(branch_name)
            else:
                branch_val[s] = self.ntuple[s].data.get_trk_feature(branch_name)

        scale_factor = {}
        for s in self.samples:
            if s == "beamoff":
                scale_factor[s] = self.ntuple["beamon"].data.trigger/self.ntuple[s].data.trigger
            else:
                scale_factor[s] = self.ntuple["beamon"].data.pot/self.ntuple[s].data.pot

        print(scale_factor)

        overlay_val_plot = []
        weight_val_plot = []
        for i in range(len(branch_val["overlay"][0])):
            overlay_val_plot.append(branch_val["overlay"][0][i])
            weight_val_plot.append(branch_val["overlay"][1][i]*scale_factor["overlay"])

        overlay_val_plot = [branch_val["beamoff"][0]] + overlay_val_plot
        weight_val_plot = [branch_val["beamoff"][1]*scale_factor["beamoff"]] + weight_val_plot
        overlay_val_plot = [branch_val["dirt"][0]] + overlay_val_plot
        weight_val_plot = [branch_val["dirt"][1]*scale_factor["dirt"]] + weight_val_plot

        weight_val_plot_fixed = [ak.where(np.isinf(w), 1, w) for w in weight_val_plot]

        # Compute histogram (not drawn yet)
        counts, bins = np.histogram(branch_val["beamon"][0], bins=bins, range=xrange)

        # Bin centers
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        # Poisson errors for counts
        errors = np.sqrt(counts)
        plt.figure(figsize=(7.0, 4.5), dpi = 200)
        # Plot as points with error bars
        plt.errorbar(bin_centers, counts, yerr=errors, fmt='o', color='black', capsize=3, label="Data")

        # (Optional) overlay step-style histogram for reference
        #plt.step(bins[:-1], counts, where='mid', color='blue', alpha=0.5, label="Histogram")
        labels = ["Dirt", "BeamOff", "Other", "pion", "proton", "muon"]
        
        plt.hist(
            overlay_val_plot,
            bins=bins,
            stacked=True,
            alpha=0.8,
            weights=weight_val_plot_fixed,
            label=labels,
            range=xrange
        )
        pot = self.ntuple[s].data.pot
        plt.title(f"POT = {pot}")
        plt.xlabel(branch_name)
        plt.ylabel("Entries per bin")
        plt.legend()
        plt.tight_layout()
        plt.show()




ccxp = dev.ccxp.ccxp(run=["run1"])
plot_branch = {
    "trk_range_muon_mom_v": (0, 1.2),
    "trk_score_v": (0.5, 1),
    "trk_llr_pid_score_v": (-1, 1),
    "trk_pid_chipr_v": (0, 500),
    "trk_distance_v": (0, 8),
    "trk_len_v": (0, 200),
    "trk_energy_proton_v": (0, 2),
    "trk_mcs_muon_mom_v": (0, 2),
    "range_mcs_difference": (-10, 2),
    "pfp_num_daughter": (0, 2)
}

for b, r in plot_branch.items():
    if b == "pfp_num_daughter":
        ccxp.plot(branch_name=b, xrange=r, bins=2, save=True)
    else:
        ccxp.plot(branch_name=b, xrange=r, save=True)

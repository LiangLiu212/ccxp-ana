import matplotlib.pyplot as plt
import numpy as np
def track_plotter(branch_name, data, mc, mc_weight, pot=0, title="title;x;y", bins=50, xrange=(-1, 1), save=False):
    # Compute histogram (not drawn yet)
    counts, bins = np.histogram(data, bins=bins, range=xrange)

    # Bin centers
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    # Poisson errors for counts
    errors = np.sqrt(counts)
    plt.figure(figsize=(6.0, 4.5), dpi = 150)
    # Plot as points with error bars
    plt.errorbar(bin_centers, counts, yerr=errors, fmt='o', color='black', capsize=3, label="Data")

    # (Optional) overlay step-style histogram for reference
    #plt.step(bins[:-1], counts, where='mid', color='blue', alpha=0.5, label="Histogram")
    labels = ["BeamOff", "Dirt", "Other", "pion", "proton", "muon"]
    hatches=["//", "", "", "", "", ""]
    colors = ["tab:gray", "#9467bd", "#d62728", "#2ca02c", "#ff7f0e", "#1f77b4"]

    plt.hist(
        mc,
        bins=bins,
        stacked=True,
        alpha=0.8,
        weights=mc_weight,
        label=labels,
        hatch=hatches,
        color=colors,
        range=xrange
    )
    plt.title(f"POT = {pot}")
    plt.xlabel(branch_name)
    plt.ylabel("Entries per bin")
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(f"/exp/uboone/app/users/liangliu/analysis-code/tutorial/script/ccxp/plot/beamon_overlay{self.runs}_{branch_name}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"/exp/uboone/app/users/liangliu/analysis-code/tutorial/script/ccxp/plot/beamon_overlay{self.runs}_{branch_name}.pdf", dpi=300, bbox_inches='tight')
    plt.show()

def event_plotter(branch_name, data, mc, mc_weight, pot=0, title="title;x;y", bins=50, xrange=(-1, 1), save=False):
    # Compute histogram (not drawn yet)
    counts, bins = np.histogram(data, bins=bins, range=xrange)

    # Bin centers
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    # Poisson errors for counts
    errors = np.sqrt(counts)
    plt.figure(figsize=(6.0, 4.5), dpi = 150)
    # Plot as points with error bars
    plt.errorbar(bin_centers, counts, yerr=errors, fmt='o', color='black', capsize=3, label="Data")

    # (Optional) overlay step-style histogram for reference
    #plt.step(bins[:-1], counts, where='mid', color='blue', alpha=0.5, label="Histogram")
    labels = ["BeamOff", "Dirt", "overlay"]
    hatches=["//", "", ""]
    colors = ["tab:gray", "#ff7f0e", "#1f77b4"]

    plt.hist(
        mc,
        bins=bins,
        stacked=True,
        alpha=0.8,
        weights=mc_weight,
        label=labels,
        hatch=hatches,
        color=colors,
        range=xrange
    )
    plt.title(f"POT = {pot}")
    plt.xlabel(branch_name)
    plt.ylabel("Entries per bin")
    plt.legend()
    plt.tight_layout()
    plt.show()

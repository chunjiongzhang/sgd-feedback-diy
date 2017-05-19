import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

sns.set_context("poster")
sns.set_style("white")
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Computer Modern"]

opt_titles = [
    ("eve", "Eve"),
    ("adam", "Adam"),
    ("adamax", "Adamax"),
    ("sgdnesterov", "SGD Nesterov"),
    ("rmsprop", "RMSprop"),
    ("sgd", "SGD"),
    ("adagrad", "Adagrad"),
    ("adadelta", "Adadelta"),
]
colors = sns.xkcd_palette(["amber", "red", "greyish", "windows blue", "faded green", "dusty purple", "black", "light blue"])

def plot_losses(model, dataset, ax, loss_ylim=None, nolrg=False):
    min_min_loss, min_max_loss = np.inf, np.inf
    for i, (opt, title) in enumerate(opt_titles):
        try:
            with open("../data/{}/{}/{}.pkl".format(model, dataset, opt), "rb") as f:
                data = pickle.load(f)
        except FileNotFoundError:
            continue
        loss_history = data["best_loss_history"]

        if .95*np.min(loss_history) < min_min_loss:
            min_min_loss = .95*np.min(loss_history)
        if .95*np.max(loss_history) < min_max_loss:
            min_max_loss = .95*np.max(loss_history)

        if nolrg is False:
            lr_str = "($\\alpha\\ 10^{{{}}}$".format(int(np.round(np.log10(data["best_opt_config"]["lr"]))))
            try:
                decay = data["best_decay"]
                if np.isclose(decay, 0):
                    decay_str = ", $\\gamma\\ 0)$"
                else:
                    decay_str = ", $\\gamma\\ 10^{{{}}})$".format(int(np.round(np.log10(decay))))
            except KeyError:
                decay_str = ")"
        else:
            lr_str = decay_str = ""
        ax.semilogy(range(1, len(loss_history) + 1), loss_history, label="{} {}{}".format(title, lr_str, decay_str), color=colors[i], linewidth=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    if loss_ylim is not None:
        ax.set_ylim(*loss_ylim)
        ax.set_yticks(np.logspace(np.log10(loss_ylim[0]), np.log10(loss_ylim[1]), 5))
    else:
        ax.set_ylim(min_min_loss, min_max_loss)
        ax.set_yticks(np.logspace(np.log10(min_min_loss), np.log10(min_max_loss), 5))
    ax_sexy_yticks = []
    for tick in ax.get_yticks():
        base, exp = "{:.1e}".format(tick).split("e")
        ax_sexy_yticks.append("${:.1f}\\times 10^{{{}}}$".format(float(base), int(exp)))
    ax.set_yticklabels(ax_sexy_yticks)
    ax.legend()

def generic_visualize(model, dataset, loss_ylim=None, nolrg=False, saveax1=None):
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    plot_losses(model, dataset, ax1, loss_ylim, nolrg)

    with open("../data/{}/{}/eve.pkl".format(model, dataset), "rb") as f:
        eve_data = pickle.load(f)
    ds = np.array([x.item() for x in eve_data["ds"]])
    eff_ds = ds * (1. + (eve_data["best_opt_config"]["decay"] * np.arange(1, len(ds) + 1)))
    
    ax2.semilogx(ds)
    ds_lims = [.9*np.min(ds), 1.1*np.max(ds)]
    ax2.set_ylim(*ds_lims)
    ax2.set_yticks(np.linspace(ds_lims[0], ds_lims[1], 3))
    ax2_pretty_yticks = []
    for tick in ax2.get_yticks():
        ax2_pretty_yticks.append("{:.1f}".format(tick))
    ax2.set_yticklabels(ax2_pretty_yticks)
    ax2.set_xticks([])
    ax2.set_ylabel("$d_t$")

    ax3.loglog(eff_ds)
    ax3.set_xlabel("$t$")
    ax3.set_ylabel("$d_t(1 + \\gamma t)$")

    fig.tight_layout()
    plt.show()
#    if saveax1 is not None:
#        extent = ax1.get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
#        fig.savefig(saveax1, bbox_inches=extent.expanded(1.02, 1.02))
        
generic_visualize("cnn", "cifar10", nolrg=True, saveax1="../data/paper_figures/cnn_cifar10.pdf")

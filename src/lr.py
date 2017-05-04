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
colors = sns.xkcd_palette(["amber", "red", "greyish", "windows blue", "faded green", "dusty purple", "black", "light blue"])

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
with open("lr.pkl", "rb") as f:
#with open("../data/batch_logistic_adams_vs_eve/mnist/eve.pkl", "rb") as f:
	evep, = plt.semilogy(pickle.load(f)["best_loss_history"][:3], label="Eve", color=colors[0], linewidth=3)
#adam_colors = sns.color_palette("Reds", 10)
#adamps = []
#for i in range(1, 3):
#for i in range(1, 11):
#    with open("../data/batch_logistic_adams_vs_eve/mnist/adam_{}.pkl".format(i), "rb") as f:
#        adamp, = plt.semilogy(pickle.load(f)["best_loss_history"][:20000], label="Adam ($\\alpha\\ {}\\times 10^{{-2}}$)".format(i), color=adam_colors[i - 1], linewidth=3)
#       adamps.append(adamp)
plt.xlabel("Epoch")
plt.ylabel("Loss")
#plt.ylim(1e-8, 1e-2)
#l1 = plt.legend(handles=[evep]+adamps[:5], loc="lower left")
#plt.gca().add_artist(l1)
#l2 = plt.legend(handles=adamps[5:], loc="upper right")

#plt.subplot(1, 2, 2)
#for i in range(1, 11):
#    for alg, alg_name, color in zip(["eve", "adam"], ["Eve", "Adam"], colors[:2]):
#        with open("../data/batch_logistic_adams_and_eves/mnist/{}_rep{}.pkl".format(alg, i), "rb") as f:
#            if i == 1:
#                plt.semilogy(pickle.load(f)["best_loss_history"], label=alg_name, color=color)
#            else:
#                plt.semilogy(pickle.load(f)["best_loss_history"], color=color)
#plt.xlabel("Epoch")
#plt.legend()

plt.tight_layout()
plt.savefig("../data/paper_figures/adam_vs_eve.pdf")
plt.show()

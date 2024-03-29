import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
import pandas as pd

##=====================================================================
typing = sys.argv[1] #'E2C'
plot_fn= 'Loss' + typing + '.png'
fig = plt.figure(figsize=(5.,3.5))

dat = pd.read_csv("../output/log_train_val" + typing + ".txt", sep="\t", header=None)
dat.columns=["Epoch","LossT", "LossV"]
t = dat["Epoch"].values
lossT = dat["LossT"].values
lossV = dat["LossV"].values


plt.xlabel('Epoch', labelpad=2)
plt.ylabel('Loss', labelpad=2)
plt.plot(t, lossT, ls='--', lw=1.2)
plt.semilogy(t, lossT, ls='-', lw=1.2)
#plt.plot(t, lossV, ls='-', lw=1.2)
plt.semilogy(t, lossV, ls='--', lw=1.2)
#plt.legend(["Train", "Validation"])
#plt.ylim(0.0, 0.01)
fig.set_tight_layout(True)
plt.show()
fig.savefig(plot_fn)
os.system("epscrop %s %s" % (plot_fn, plot_fn))

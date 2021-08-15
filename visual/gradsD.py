import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import random

##=====================================================================
dat = pd.read_csv("../output/gradsD.txt", sep="\t")
layers = list(dat.columns)
n = len(layers)
st = ["-", "--", "-.",":"]*n
random.seed(11)
ls = random.sample(st,n)

dat.plot(logy=True, figsize=(10,6), style = ls)
plt.xlabel('Epochs(10x)', labelpad=2)
plt.ylabel('Grads.Sum', labelpad=2)
plt.legend(loc="upper right", ncol = n//4, frameon=False)
plt.show()
plt.savefig('gradsD.png')

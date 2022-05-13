import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd

from os.path import join

experiment = "pendulum-51"

path = join(os.path.abspath("./"), 'doctors', experiment)
hyper = os.path.isfile(join(path, "results.csv"))

print(path)

if not hyper:
    print(f"No hyper optimization was found for [{experiment}]")
    exit()

df = pd.read_csv(os.path.join(path, "results.csv"))
n_ivs = len(df.columns) - 1
ivs = list(df.columns.values)
del ivs[ivs.index("nrmse")]
dv ="nrmse"

if n_ivs != 2:
     print("Not 3d")
     exit()

fig = plt.figure(figsize=(7, 7))
ax = plt.axes(projection='3d')

ax.scatter3D(df[ivs[0]], df[ivs[1]], df[dv], c=df[dv], cmap="Reds")
plt.show()


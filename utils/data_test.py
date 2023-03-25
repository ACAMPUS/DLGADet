import json
import pylab as pl
import random
import numpy as np
import cv2
import anno_func



datadir = r"D:\dataset\data"

filedir = datadir + "/annotations.json"
ids = open(datadir + "/test/ids.txt").read().splitlines()

annos = json.loads(open(filedir).read())
annos.keys()

imgid = random.sample(ids, 1)[0]
print(imgid)

imgdata = anno_func.load_img(annos, datadir, imgid)
imgdata_draw = anno_func.draw_all(annos, datadir, imgid, imgdata)
pl.figure(figsize=(20,20))
pl.imshow(imgdata_draw)
pl.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2)
# ax = axes.ravel()
data = pd.Series(np.random.rand(16), index=list('abcdefghijklmnop'))

data.plot.bar(ax=axes[1, 0], color='b', alpha=0.5)
data.plot.barh(ax=axes[0, 1], color='k', alpha=0.5)

plt.show()
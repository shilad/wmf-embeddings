import logging
import os.path
import numpy as np
import pandas as pd
import scipy.spatial.distance

from utils import Titler, nearest_neighbors

TESTS = [
    'c:155',
    'c:152',
    'c:144',
    'c:7346',
    'simple.wikipedia:dog',
    'simple.wikipedia:man',
    'simple.wikipedia:woman',
    'simple.wikipedia:jazz'
]

ids = [line.strip() for line in open('output/joint_ids.txt', encoding='utf-8')]
titler = Titler('./output/titles.csv')

for i in range(10):
    print('testing epoch %d' % i)
    e = np.load('output/joint_vectors.%d.npy' % i)
    for w in TESTS:
        if w not in ids: continue
        i = ids.index(w)
        v = e[i,:]
        print('\n\n\nneighbors for %s:' % titler.get_title(w))
        for index in nearest_neighbors(e, v, 20):
            print('\t%s' % titler.get_title(ids[index]))

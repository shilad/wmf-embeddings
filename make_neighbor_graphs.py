#!/usr/bin/python3 -O
#
# Builds neighbor graphs for every language edition of Wikipedia.
#

import multiprocessing
import logging
import os.path

import sys
from dynarray import DynamicArray
from scipy.sparse import csr_matrix, save_npz

from utils import Titler, LangEmbedding, max_cores, wiki_dirs


def main(dir):
    global titler

    titler = Titler(os.path.join(dir, 'titles.csv'))

    with multiprocessing.Pool(max_cores()) as pool:
        pool.map(make_neighbors, wiki_dirs(dir))

def make_neighbors(path):
    global titler

    lang = os.path.basename(path)

    emb = LangEmbedding(lang, path, titler)
    emb.build_fast_knn()

    n = emb.nrows()
    rows = DynamicArray(dtype='int32')
    cols = DynamicArray(dtype='int32')
    vals = DynamicArray(dtype='float32')

    for i, id in enumerate(emb.ids):
        if i % 10000 == 0:
            logging.info('generating neighbors for id %d of %d in %s', i, len(emb.ids), emb.lang)
        neighbors = emb.neighbors(id, n=100, include_distances=True, use_indexes=True)
        for (j, dist) in neighbors:
            if j != i:
                rows.append(i)
                cols.append(j)
                vals.append(dist)

    csr = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype='float32')
    save_npz(os.path.join(emb.dir, 'neighbors'), csr)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        sys.stderr.write('usage: %s path/to/dir' % sys.argv[0])
        sys.exit(1)
    main(sys.argv[1])
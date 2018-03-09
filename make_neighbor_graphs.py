#!/usr/bin/python -O

from collections import defaultdict
import logging
import os.path

from dynarray import DynamicArray
from scipy.sparse import csr_matrix, save_npz

from utils import Titler, LangEmbedding

def main(dir):
    titler = Titler(os.path.join(dir, 'titles.csv'))
    embeddings = []

    for lang in os.listdir(dir):
        p = os.path.join(dir, lang)
        if os.path.isdir(p) and lang.endswith('wiki'):
            emb = LangEmbedding(lang, p, titler, aligned=True)
            emb.build_fast_knn()
            embeddings.append(emb)

    for emb in embeddings:
        n = emb.nrows()
        rows = DynamicArray(dtype='int32')
        cols = DynamicArray(dtype='int32')
        vals = DynamicArray(dtype='float32')

        for i, id in enumerate(emb.ids):
            if i % 10000 == 0:
                logging.info('generating neighbors for id %d of %d in %s', i, len(emb.ids), emb.lang)
            neighbors = emb.neighbors(id, n=50, include_distances=True, use_indexes=True)
            for (j, dist) in neighbors:
                if j != i:
                    rows.append(i)
                    cols.append(j)
                    vals.append(dist)

        csr = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype='float32')
        save_npz(os.path.join(emb.dir, 'neighbors'), csr)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main('./output')
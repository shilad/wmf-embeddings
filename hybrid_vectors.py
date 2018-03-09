
import logging
import math
import numpy as np
import os.path
import pandas as pd
import scipy.linalg
import scipy.spatial.distance
from sklearn.decomposition import TruncatedSVD

from collections import defaultdict

from utils import LangEmbedding, Titler, nearest_neighbors, NP_FLOAT, NeighborGraph

SAMPLE_SIZE = 10000

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

def show_examples(matrix, ids, titler):
    for t in TESTS:
        if t not in ids: continue
        i = ids.index(t)
        neighbor_indexes = nearest_neighbors(matrix, matrix[i,:], 5)
        neighbor_titles = [titler.get_title(ids[i]) for i in neighbor_indexes]
        print('results for %s: %s' % (titler.get_title(t), ', '.join(neighbor_titles)))

NEIGHBOR_WEIGHT = 0.3     # weight of neighbors compared to original vector
MATCHED_MULTIPLIER = 2    # multiplier for neighbor vectors that match.

def main(path, lang):
    titler = Titler(os.path.join(path, 'titles.csv'))

    # Read in the embeddings
    nav_emb= LangEmbedding('nav', os.path.join(path, 'nav'), titler)
    lang_emb= LangEmbedding(lang, os.path.join(path, lang), titler)
    ids = sorted(set(lang_emb.ids).intersection(nav_emb.ids))

    (w2, nav_sub) = nav_emb.submatrix(ids)
    (w1, lang_sub) = lang_emb.submatrix(ids)

    M = np.concatenate([nav_sub, lang_sub], axis=1)
    logging.info("calcuating SVD of joint %d x %d matrix", M.shape[0], M.shape[1])
    hybrid = TruncatedSVD(100).fit_transform(M)

    # Calculate the alignment between each original matrix and the embedding.
    ortho, scale = scipy.linalg.orthogonal_procrustes(lang_sub, hybrid, check_finite=True)
    aligned = lang_emb.embedding.dot(ortho)

    in_indexes = lang_emb.indexes(ids)
    aligned[in_indexes] = hybrid
    in_indexes = set(in_indexes)
    out_indexes = [i for i in range(lang_emb.nrows()) if i not in in_indexes]

    # Retrofit the out of sample points
    show_examples(aligned, lang_emb.ids, titler)
    graph = NeighborGraph(lang_emb.ids, os.path.join(path, lang, 'neighbors.npz'))
    for epoch in range(10):
        change = 0.0
        for i in out_indexes:
            neighbor_indexes, weights = graph.index_neighbors_and_weights(i, 10)
            if np.sum(weights) > 0:
                weights /= np.sum(weights)
                v1 = aligned[i,:]
                v1_orig = v1.copy()
                v1 *= (1.0 - NEIGHBOR_WEIGHT)
                v1 += NEIGHBOR_WEIGHT * np.average(aligned[neighbor_indexes,:], weights=weights, axis=0)
                change += np.sum((v1 - v1_orig) ** 2) ** 0.5
        logging.info('Epoch %d: avg_change=%.4f',
                     epoch, change / len(out_indexes))

        show_examples(aligned, lang_emb.ids, titler)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main('./output', 'simplewiki')
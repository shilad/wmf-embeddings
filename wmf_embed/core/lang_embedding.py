import logging
import math
import os.path
import re

import annoy
import numpy as np
from gensim.models import KeyedVectors

from .utils import NP_FLOAT

def from_mikolov(lang, inpath, outpath):
    v = KeyedVectors.load_word2vec_format(inpath, binary=False)  # C text format
    if not os.path.isdir(outpath): os.makedirs(outpath)
    with open(outpath + '/ids.txt', 'w', encoding='utf-8') as f:
        for id in v.index2word:
            f.write(lang + '.wikipedia:')
            f.write(id)
            f.write('\n')
    with open(outpath + '/titles.csv', 'w', encoding='utf-8') as f:
        pass
    np.save(outpath + '/vectors.npy', np.array(v.vectors))

    return LangEmbedding(lang, outpath)


MIN_SIZE = 10000

class LangEmbedding(object):
    def __init__(self, lang, dir, titles=None, aligned=False, model_name=None):
        logging.info('initializing embedding in %s for %s', dir, lang)

        if model_name:
            pass    # use it
        elif aligned:
            model_name = 'vectors.aligned.npy'
        else:
            model_name = 'vectors.npy'

        self.lang = lang
        self.dir = dir
        self.ids = []
        self.titles = titles
        path_id = os.path.join(dir, 'ids.txt')
        with open(path_id, 'r', encoding='utf-8') as f:
            for line in f:
                self.ids.append(line.strip())
        self.ids_to_index = {id: i for i, id in enumerate(self.ids)}

        self.vector_path = os.path.join(dir, model_name)

        self.embedding = np.load(self.vector_path)
        self.embedding /= np.linalg.norm(self.embedding, axis=1)[:,None]    # Unit vectors
        N = self.embedding.shape[0]
        pop_weight = max(1, math.log(N / MIN_SIZE))
        self.pop = 1.0 * pop_weight / (np.arange(N) + 1)
        self.index = None   # fast knn index

    def submatrix(self, ids):
        weights = np.zeros(len(ids), dtype=NP_FLOAT)
        sub = np.zeros((len(ids), self.dims()), dtype=NP_FLOAT)
        for i, id in enumerate(ids):
            if id in self.ids_to_index:
                index = self.ids_to_index[id]
                sub[i, :] = self.embedding[index, :]
                weights[i] = self.pop[index]
        return weights, sub

    def indexes(self, ids):
        return [self.ids_to_index[id] for id in ids ]

    def dims(self):
        return self.embedding.shape[1]

    def nrows(self):
        return self.embedding.shape[0]

    def build_fast_knn(self):
        index = annoy.AnnoyIndex(self.dims(), metric='angular')

        ann_path = self.vector_path + '.ann'
        if (os.path.isfile(ann_path)
        and os.path.getmtime(ann_path) >= os.path.getmtime(self.vector_path)):
            logging.info('loading accelerated knn tree from %s', ann_path)
            index.load(ann_path)
            self.index = index
            return

        logging.info('building accelerated knn tree')

        # Build the approximate-nearest-neighbor index
        for i in range(self.embedding.shape[0]):
            index.add_item(i, self.embedding[i,:])

        index.build(10)
        index.save(self.vector_path + '.ann')
        self.index = index

    def popularity(self):
        return self.pop

    def map(self, basis):
        self.embedding.dot(basis, out=self.embedding)

    def neighbors(self, id, n=5, include_distances=False, use_indexes=True):
        assert(self.index)
        if id not in self.ids_to_index: return []
        i = self.ids_to_index[id]
        indexes, dists = self.index.get_nns_by_item(i, n, include_distances=True)
        if use_indexes:
            result = indexes
        else:
            result = [self.ids[j] for j in indexes]
        if include_distances:
            return list(zip(result, dists))
        else:
            return result

    def dense_words(self):
        word_to_sparse_index = {}
        MATCH_WORD = re.compile(r'^.*?\.wikipedia:(.*)$').match
        for (id, i) in self.ids_to_index.items():
            m = MATCH_WORD(id)
            if m:
                word = m.group(1)
                word.replace('_', ' ')
                word_to_sparse_index[word] = i

        sparse_indexes = np.sort(list(word_to_sparse_index.values()))
        sparse_to_dense = { s : d for d, s in enumerate(sparse_indexes) }

        word_matrix = self.embedding[sparse_indexes, :]
        word2id = { w : sparse_to_dense[word_to_sparse_index[w]] for w in word_to_sparse_index }

        return word2id, word_matrix
import logging
import annoy
import math
import numpy as np
import os.path
import pandas as pd
import re
import scipy.spatial.distance

from scipy.sparse import coo_matrix, csr_matrix, load_npz, save_npz
from collections import defaultdict
from dynarray import DynamicArray

NP_FLOAT = 'float32'

MIN_SIZE = 10000

class LangEmbedding(object):
    def __init__(self, lang, dir, titles, aligned=False, model_name=None):
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


def read_embeddings(titler, path, aligned=False):
    dims = None
    embeddings = []
    for wiki in os.listdir(path):
        subdir = os.path.join(path, wiki)
        if os.path.isdir(subdir) and (wiki == 'nav' or 'wiki' in wiki):
            lang = LangEmbedding(wiki, subdir, titler, aligned=aligned)
            if dims is None:
                dims = lang.dims()
            else:
                assert(lang.dims() == dims)
            embeddings.append(lang)

    return embeddings


class Titler(object):
    def __init__(self, path):
        logging.info('reading titles from %s', path)
        titles_df = pd.read_csv(path)
        self.titles = {}
        for i, row in titles_df.iterrows():
            if row['project'] == 'concept':
                self.titles['c:' + str(row['page_id'])] = row['title']
        logging.info('reading %d titles', len(self.titles))

    def get_title(self, id):
        if id in self.titles:
            return self.titles[id]
        else:
            return id.split(':', 1)[-1]


class NeighborGraph(object):
    def __init__(self, ids, path=None):
        self.ids = ids
        self.ids_to_indexes = { id : i for i, id in enumerate(ids) }
        self.path = path

        n = len(self.ids)
        if path:
            logging.info('loading neighbors from %s', path)
            self.graph = load_npz(path)
        else:
            self.graph = csr_matrix((n, n))


    def merge(self, otherGraph):
        self.graph = self.graph + otherGraph.graph

    def neighbors(self, id):
        if id not in self.ids_to_indexes:
            return set()
        row = self.graph[self.ids_to_indexes[id],:] # row is sparse!
        rows, cols = row.nonzero()
        return set([self.ids[i] for i in cols])

    def index_neighbors(self, index):
        row = self.graph[index,:] # row is sparse!
        return row.nonzero()[1]     # columns

    def index_neighbors_and_weights(self, index, n=10):
        coo = self.graph.getrow(index).tocoo()
        indexes = np.argsort(coo.data)[:n]
        # angular distance is sqrt(2 * (1 - cos(u,v)))
        # we solve for cos below as 1 - ang-dist**2 / 2
        weights = (1 - coo.data ** 2 / 2)[indexes]

        # As an ad-hoc thing, it's good to make the weights drop off more quickly
        return coo.col[indexes], weights * weights

    def save_npz(self, path):
        save_npz(path, self.graph)

    def num_edges(self):
        return len(self.graph.nonzero()[0])

    def num_nodes(self):
        return len(self.nodes())

    def nodes(self):
        return [self.ids[i] for i in set(self.graph.nonzero()[0])]

def wiki_to_project(wiki):
    if wiki.endswith('wiki'):
        return wiki[:-4] + '.wikipedia'
    else:
        return wiki + '.wikipedia'

def project_to_wiki(project):
    if project.endswith('.wikipedia'):
        return project.split('.')[0] + 'wiki'
    else:
        return project


def max_cores():
    return min(max(1, os.cpu_count()), 8)


def nearest_neighbors(matrix, vector, n):
    v = vector.reshape(1, -1)
    dists = scipy.spatial.distance.cdist(matrix, v, 'cosine').reshape(-1)
    indexes = np.argsort(dists)
    return indexes.tolist()[:n]

def wiki_dirs(path, include_nav=False):
    result = []

    for fn in os.listdir(path):
        full_path = os.path.join(path, fn)
        if os.path.isdir(full_path):
            if include_nav and fn == 'nav':
                result.append(full_path)
            elif fn.endswith('wiki'):
                result.append(full_path)

    return result

def read_ids(path):
    with open(path, encoding='utf-8') as f:
        return [ line.strip() for line in f ]
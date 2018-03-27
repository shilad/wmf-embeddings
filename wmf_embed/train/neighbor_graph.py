import logging

import numpy as np
from scipy.sparse import load_npz, csr_matrix, save_npz


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
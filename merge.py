import logging
from collections import defaultdict

import os.path
import numpy as np

from utils import LangEmbedding, Titler, read_embeddings, NP_FLOAT, NeighborGraph, wiki_to_project


def main(path):
    titler = Titler(os.path.join(path, 'titles.csv'))

    embeddings = read_embeddings(titler, path, aligned=True)

    id_sums = defaultdict(float)
    for emb in embeddings:
        for (id, pop) in zip(emb.ids, emb.popularity().tolist()):
            id_sums[id] += pop

    ids = sorted(id_sums.keys())
    ids_to_indexes = {id : i for i, id in enumerate(ids) }

    joint = np.zeros((len(id_sums), embeddings[0].dims()), dtype=NP_FLOAT)
    for emb in embeddings:
        for i, (id, pop) in enumerate(zip(emb.ids, emb.popularity().tolist())):
            joint[ids_to_indexes[id],:] += pop / id_sums[id] * emb.embedding[i,:]

    np.save(os.path.join(path, 'joint_vectors'), joint)
    with open(os.path.join(path, 'joint_ids.txt'), 'w', encoding='utf-8') as f:
        for id in ids:
            f.write(id)
            f.write('\n')

    # Merge neighborhoods
    projs = set([wiki_to_project(e.lang) for e in embeddings])
    graph_paths = [path + '/word_neighbors.tsv'] + [ e.dir + '/neighbors.tsv' for e in embeddings ]
    neighbors = NeighborGraph(ids, projs=projs)
    for path in graph_paths:
        g = NeighborGraph(ids, path, projs=projs)
        logging.info('neighbor graph %s: read %d nodes %d edges',
                      g.path, g.num_nodes(), g.num_edges())
        neighbors.merge(g)
    logging.info('merged neighbor graph: %d nodes %d edges',
                 neighbors.num_nodes(), neighbors.num_edges())
    neighbors.save_npz(path + '/neighbors.npz')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main('./output')
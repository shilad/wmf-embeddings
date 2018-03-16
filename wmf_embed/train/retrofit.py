import logging
import numpy as np

from wmf_embed.train.utils import NeighborGraph

NEIGHBOR_WEIGHT = 0.3     # weight of neighbors compared to original vector

def main(path):
    # titler = Titler(path + '/titles.csv')
    joint_ids = [token.strip() for token in open(path + '/joint_ids.txt', encoding='utf-8')]
    embedding = np.load(path + '/joint_vectors.npy')
    neighbors = NeighborGraph(joint_ids, npz_path=(path + '/neighbors.npz'))

    for epoch in range(10):
        logging.info('Beginning retrofitting epoch %d', epoch)
        retrofit(epoch, embedding, neighbors, None)
        np.save(path + '/joint_vectors.' + str(epoch) + '.npy', embedding)



def retrofit(epoch, embedding, neighbors, titler):
    test(embedding, )
    change = 0.0
    indexes = np.arange(neighbors.num_nodes())
    np.random.shuffle(indexes)
    for node_num, i in enumerate(indexes):
        if node_num % 10000 == 0:
            logging.info('Epoch %d: retrofitting %d of %d, avg_change=%.4f',
                         epoch, node_num, len(indexes), change / (node_num + 1))

        neighbor_indexes = neighbors.index_neighbors(i)
        if len(neighbor_indexes) >= 1:
            v1 = embedding[i,:]
            v1_orig = v1.copy()
            v1 *= (1.0 - NEIGHBOR_WEIGHT)
            v1 += NEIGHBOR_WEIGHT * np.mean(embedding[neighbor_indexes,:], axis=0)
            change += np.sum((v1 - v1_orig) ** 2) ** 0.5




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main('./output')
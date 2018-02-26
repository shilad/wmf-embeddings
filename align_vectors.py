
import logging
import math
import numpy as np
import os.path
import pandas as pd
import scipy.linalg
import scipy.spatial.distance

from collections import defaultdict

SAMPLE_SIZE = 10000

TESTS = [
    'c:155',
    'c:152',
    'c:144',
    'c:7346',
    'dog',
    'jazz'
]

class LangEmbedding(object):
    def __init__(self, lang, dir, titles):
        logging.info('initializing embedding in %s for %s', dir, lang)

        self.lang = lang
        self.ids = []
        self.titles = titles
        path_id = os.path.join(dir, 'ids.txt')
        with open(path_id, 'r', encoding='utf-8') as f:
            for line in f:
                self.ids.append(line.strip())

        self.embedding = np.load(os.path.join(dir, 'vectors.npy'))
        self.embedding /= np.linalg.norm(self.embedding, axis=1)[:,None]    # Unit vectors
        self.pop = 1.0 / (np.arange(self.embedding.shape[0]) + 1)

    def submatrix(self, ids):
        id_to_index = { id: i for (i, id) in enumerate(self.ids) }
        weights = np.zeros(len(ids))
        sub = np.zeros((len(ids), self.dims()))
        for i, id in enumerate(ids):
            if id in id_to_index:
                index = id_to_index[id]
                sub[i, :] = self.embedding[index, :]
                weights[i] = self.pop[index]
        return weights, sub

    def dims(self):
        return self.embedding.shape[1]

    def popularity(self):
        return self.pop

    def map(self, basis):
        self.embedding.dot(basis, out=self.embedding)

    def show_examples(self):
        return show_examples(self.embedding, self.ids, self.titles)


def show_examples(matrix, ids, titles):
    for t in TESTS:
        if t not in ids: continue
        i = ids.index(t)
        neighbor_indexes = nearest_neighbors(matrix, matrix[i,:], 5)
        neighbor_titles = [get_title(ids[i], titles) for i in neighbor_indexes]
        print('results for %s: %s' % (get_title(t, titles), ', '.join(neighbor_titles)))


def get_title(id, titles):
    if id in titles:
        return titles[id]
    else:
        return id.split(':', 1)[-1]

def main(path):
    titles = read_titles(os.path.join(path, 'titles.csv'))

    # Read in the embeddings
    nav = LangEmbedding('nav', os.path.join(path, 'nav'), titles)
    langs = []
    for wiki in os.listdir(path):
        subdir = os.path.join(path, wiki)
        if os.path.isdir(subdir) and wiki != 'nav':
            lang = LangEmbedding(wiki, subdir, titles)
            assert(lang.dims() == nav.dims())
            langs.append(lang)

    # Choose the ids
    sample_ids, sample_probs = create_sample_distribution([nav] + langs)

    align([nav] + langs, sample_ids, sample_probs, titles)


def read_titles(path):
    titles_df = pd.read_csv(path)
    titles = {}
    for i, row in titles_df.iterrows():
        if row['project'] == 'concept':
            titles['c:' + str(row['page_id'])] = row['title']
    return titles

def create_sample_distribution(embeddings):
    id_scores = defaultdict(float)
    id_counts = defaultdict(int)
    for emb in embeddings:
        for (id, pop) in zip(emb.ids, emb.popularity()):
            id_scores[id] += pop
            id_counts[id] += 1

    ids = []
    scores = []
    for (id, score) in id_scores.items():
        n = id_counts[id]
        if n >= 2:
            ids.append(id)
            scores.append(score * math.log(n))

    ids = np.array(ids)
    scores = np.array(scores)
    scores = scores / scores.sum()

    return (ids, scores)

def align(embeddings, ids, sample_probs, titles):
    emb_common = []
    for emb in embeddings:
        w, M = emb.submatrix(ids)
        emb_common.append((w, M))

    for i in range(20):
        # Calculate the sample matrices
        if len(ids) <= SAMPLE_SIZE:
            indexes = np.arange(len(ids))
        else:
            indexes = np.random.choice(len(ids), SAMPLE_SIZE, replace=False, p=sample_probs)

        # Calculate the samples
        emb_samples = []
        for w, M in emb_common:
            emb_samples.append((w[indexes], M[indexes, :]))

        # calculate the weighted average embedding
        consensus = np.zeros((len(indexes), embeddings[0].dims()))
        sample_weights = np.zeros(len(indexes))
        for w, M in emb_samples:
            consensus += M * w[:, None]
            sample_weights += w
        consensus /= sample_weights[:, None]

        show_examples(consensus, [ids[i] for i in indexes], titles)

        # calculate the err2
        errSum = 0
        weightSum = 0
        for w, M in emb_samples:
            errs = np.sum((M - consensus) ** 2, axis=1) ** 0.5
            errSum += np.sum(w * errs)
            weightSum += np.sum(w)
        err = errSum / weightSum
        logging.info('average L2 error at iteration %d is %.3f', i, err)

        # calculate and apply the transformations
        for i in range(len(emb_samples)):
            w, M = emb_samples[i]
            mask = (w > 0.0)
            ortho, scale = scipy.linalg.orthogonal_procrustes(M[mask], consensus[mask], check_finite=True)
            w2, M2 = emb_common[i]
            M2.dot(ortho, out=M2)


def nearest_neighbors(matrix, vector, n):
    v = vector.reshape(1, -1)
    dists = scipy.spatial.distance.cdist(matrix, v, 'cosine').reshape(-1)
    indexes = np.argsort(dists)
    return indexes.tolist()[:n]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main('./output')

import logging
import math
import numpy as np
import os.path
import pandas as pd
import scipy.linalg
import scipy.spatial.distance

from collections import defaultdict

from utils import LangEmbedding, Titler, nearest_neighbors, NP_FLOAT

SAMPLE_SIZE = 10000

TESTS = [
    'c:155',
    'c:152',
    'c:144',
    'c:7346',
    'dog',
    'jazz'
]

def show_examples(matrix, ids, titler):
    for t in TESTS:
        if t not in ids: continue
        i = ids.index(t)
        neighbor_indexes = nearest_neighbors(matrix, matrix[i,:], 5)
        neighbor_titles = [titler.get_title(ids[i]) for i in neighbor_indexes]
        print('results for %s: %s' % (titler.get_title(t), ', '.join(neighbor_titles)))


def get_title(id, titles):
    if id in titles:
        return titles[id]
    else:
        return id.split(':', 1)[-1]

def main(path):
    titler = Titler(os.path.join(path, 'titles.csv'))

    # Read in the embeddings
    nav = LangEmbedding('nav', os.path.join(path, 'nav'), titler)
    langs = []
    for wiki in os.listdir(path):
        subdir = os.path.join(path, wiki)
        if os.path.isdir(subdir) and wiki != 'nav':
            lang = LangEmbedding(wiki, subdir, titler)
            assert(lang.dims() == nav.dims())
            langs.append(lang)

    # Choose the ids
    sample_ids, sample_probs = create_sample_distribution([nav] + langs)

    # Align the vectors
    align([nav] + langs, sample_ids, sample_probs, titler)

    # Write the aligned results
    for emb in [nav] + langs:
        np.save(os.path.join(path, emb.lang, 'vectors.aligned'), emb.embedding)


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
    scores = np.array(scores, dtype=NP_FLOAT)
    scores = scores / scores.sum()

    return (ids, scores)

def align(embeddings, ids, sample_probs, titler):
    emb_common = []
    for emb in embeddings:
        w, M = emb.submatrix(ids)
        emb_common.append((w, M, np.identity(emb.dims(), dtype=NP_FLOAT)))

    for i in range(20):
        # Calculate the sample matrices
        if len(ids) <= SAMPLE_SIZE:
            indexes = np.arange(len(ids))
        else:
            indexes = np.random.choice(len(ids), SAMPLE_SIZE, replace=False, p=sample_probs)

        # Calculate the samples
        emb_samples = []
        for w, M, rot in emb_common:
            emb_samples.append((w[indexes], M[indexes, :]))

        # calculate the weighted average embedding
        consensus = np.zeros((len(indexes), embeddings[0].dims()), dtype=NP_FLOAT)
        sample_weights = np.zeros(len(indexes), dtype=NP_FLOAT)
        for w, M in emb_samples:
            consensus += M * w[:, None]
            sample_weights += w
        consensus /= sample_weights[:, None]

        show_examples(consensus, [ids[i] for i in indexes], titler)

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
            w2, M2, rot = emb_common[i]
            M2.dot(ortho, out=M2)
            rot.dot(ortho, out=rot)

    # apply the final rotations to the full matrices
    for emb, (w, M, rot) in zip(embeddings, emb_common):
        emb.map(rot)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main('./output')
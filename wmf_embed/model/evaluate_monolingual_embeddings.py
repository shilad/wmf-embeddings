#!/usr/bin/env python3 -O
#
# Script to evaluate embeddings.
#
# Based on the evaluation script for MUSE: https://github.com/facebookresearch/MUSE
#
import logging
import sys

import os
import numpy as np

from scipy.stats import spearmanr

from ..core.utils import wiki_dirs
from ..core.titler import Titler
from ..core.lang_embedding import LangEmbedding, from_mikolov

MONOLINGUAL_EVAL_PATH = 'muse-eval/monolingual'


def get_word_pairs(path, lower=True):
    """
    Return a list of (word1, word2, score) tuples from a word similarity file.
    """
    assert os.path.isfile(path) and type(lower) is bool
    word_pairs = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            line = line.lower() if lower else line
            line = line.split()
            # ignore phrases, only consider words
            if len(line) != 3:
                assert len(line) > 3
                assert 'SEMEVAL17' in os.path.basename(path) or 'EN-IT_MWS353' in path
                continue
            word_pairs.append((line[0], line[1], float(line[2])))
    return word_pairs


def get_word_id(word, word2id, lower):
    """
    Get a word ID.
    If the model does not use lowercase and the evaluation file is lowercased,
    we might be able to find an associated word.
    """
    assert type(lower) is bool
    word_id = word2id.get(word)
    if word_id is None and not lower:
        word_id = word2id.get(word.capitalize())
    if word_id is None and not lower:
        word_id = word2id.get(word.title())
    return word_id


def get_spearman_rho(word2id1, embeddings1, path, lower,
                     word2id2=None, embeddings2=None):
    """
    Compute monolingual or cross-lingual word similarity score.
    """
    assert not ((word2id2 is None) ^ (embeddings2 is None))
    word2id2 = word2id1 if word2id2 is None else word2id2
    embeddings2 = embeddings1 if embeddings2 is None else embeddings2
    assert len(word2id1) == embeddings1.shape[0]
    assert len(word2id2) == embeddings2.shape[0]
    assert type(lower) is bool
    word_pairs = get_word_pairs(path)
    not_found = 0
    pred = []
    gold = []
    for word1, word2, similarity in word_pairs:
        id1 = get_word_id(word1, word2id1, lower)
        id2 = get_word_id(word2, word2id2, lower)
        if id1 is None or id2 is None:
            not_found += 1
            continue
        u = embeddings1[id1]
        v = embeddings2[id2]
        score = u.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v))
        gold.append(similarity)
        pred.append(score)
    return spearmanr(gold, pred).correlation, len(gold), not_found


def get_wordsim_scores(language, word2id, embeddings, lower=True):
    """
    Return monolingual word similarity scores.
    """
    dirpath = os.path.join(MONOLINGUAL_EVAL_PATH, language)
    if not os.path.isdir(dirpath):
        return None

    scores = {}
    separator = "=" * (30 + 1 + 10 + 1 + 13 + 1 + 12)
    pattern = "%30s %10s %13s %12s"
    logging.info(separator)
    logging.info(pattern % ("Dataset", "Found", "Not found", "Rho"))
    logging.info(separator)

    total = [0, 0, 0, 0]
    for filename in list(os.listdir(dirpath)):
        if filename.startswith('%s_' % (language.upper())):
            filepath = os.path.join(dirpath, filename)
            coeff, found, not_found = get_spearman_rho(word2id, embeddings, filepath, lower)
            logging.info(pattern % (filename[:-4], str(found), str(not_found), "%.4f" % coeff))
            total[0] += found
            total[1] += not_found
            total[2] += coeff
            total[3] += 1
            scores[filename[:-4]] = coeff

    logging.info(pattern % ('OVERALL', str(total[0]), str(total[1]), "%.4f" % (total[2] / total[3])))
    logging.info(separator)
    logging.info('')

    return scores


def get_wordanalogy_scores(language, word2id, embeddings, lower):
    """
    Return (english) word analogy score
    """
    dirpath = os.path.join(MONOLINGUAL_EVAL_PATH, language)
    if not os.path.isfile(os.path.join(dirpath, 'questions-words.txt')):
        return

    assert type(lower) is bool

    # normalize word embeddings
    embeddings = embeddings / np.sqrt((embeddings ** 2).sum(1))[:, None]

    # scores by category
    scores = {}

    word_ids = {}
    queries = {}

    for line in open(os.path.join(dirpath, 'questions-words.txt'), 'r'):

        # new line
        line = line.rstrip()
        if lower:
            line = line.lower()

        # new category
        if ":" in line:
            assert line[1] == ' '
            category = line[2:]
            assert category not in scores
            scores[category] = {'n_found': 0, 'n_not_found': 0, 'n_correct': 0}
            word_ids[category] = []
            queries[category] = []
            continue

        # get word IDs
        assert len(line.split()) == 4, line
        word1, word2, word3, word4 = line.split()
        word_id1 = get_word_id(word1, word2id, lower)
        word_id2 = get_word_id(word2, word2id, lower)
        word_id3 = get_word_id(word3, word2id, lower)
        word_id4 = get_word_id(word4, word2id, lower)

        # if at least one word is not found
        if any(x is None for x in [word_id1, word_id2, word_id3, word_id4]):
            scores[category]['n_not_found'] += 1
            continue
        else:
            scores[category]['n_found'] += 1
            word_ids[category].append([word_id1, word_id2, word_id3, word_id4])
            # generate query vector and get nearest neighbors
            query = embeddings[word_id1] - embeddings[word_id2] + embeddings[word_id4]
            query = query / np.linalg.norm(query)

            queries[category].append(query)

    # Compute score for each category
    for cat in queries:
        qs = np.vstack(queries[cat])
        keys = embeddings.T
        values = np.matmul(qs, keys)

        # be sure we do not select input words
        for i, ws in enumerate(word_ids[cat]):
            for wid in [ws[0], ws[1], ws[3]]:
                values[i, wid] = -1e9

        scores[cat]['n_correct'] = np.sum(values.argmax(axis=1) == [ws[2] for ws in word_ids[cat]])

    # pretty print
    separator = "=" * (30 + 1 + 10 + 1 + 13 + 1 + 12)
    pattern = "%30s %10s %13s %12s"
    logging.info(separator)
    logging.info(pattern % ("Category", "Found", "Not found", "Accuracy"))
    logging.info(separator)

    # compute and log accuracies
    total = [0, 0, 0, 0]
    accuracies = {}
    for k in sorted(scores.keys()):
        v = scores[k]
        accuracies[k] = float(v['n_correct']) / max(v['n_found'], 1)
        logging.info(pattern % (k, str(v['n_found']), str(v['n_not_found']), "%.4f" % accuracies[k]))

        total[0] += v['n_found']
        total[1] += v['n_not_found']
        total[2] += accuracies[k]
        total[3] += 1

    logging.info(pattern % ('OVERALL', str(total[0]), str(total[1]), "%.4f" % (total[2] / total[3])))
    logging.info(separator)
    logging.info('')

    return accuracies


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate a language model.')
    parser.add_argument('--path', required=True, help='full path to model')
    parser.add_argument('--lang', required=True, help='language code of model (e.g. en)')
    parser.add_argument('--lower', action='store_true', default=False, help='whether to lowercase words in the corpus')

    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    if args.lang == 'simple': args.lang = 'en'  # hack

    logging.basicConfig(format='%(asctime)s ' + args.path + ': %(message)s', level=logging.INFO)

    mono_langs = set(os.listdir(MONOLINGUAL_EVAL_PATH))
    if args.lang not in mono_langs:
        logging.error('No monolingual evaluations for %s; exiting.', args.lang)
        sys.exit(1)

    embed = from_mikolov(args.lang, args.path, args.path + '.wmf')
    word2id, matrix = embed.dense_words()
    if not args.lower:
        numCaseEqual = 0
        numWords = 0
        for w in word2id:
            if not w.startswith('t:'):
                numWords += 1
                if w == w.lower() or w == w.casefold():
                    numCaseEqual += 1
        if numCaseEqual / numWords > 0.999:
            args.lower = True
            logging.warning('Case folding not requested but applying case folding because '
                            '%.4f%% of vocabulary is case insensitive.',
                            100.0 * numCaseEqual / numWords)


    get_wordsim_scores(args.lang, word2id, matrix)
    get_wordanalogy_scores(args.lang, word2id, matrix, args.lower)

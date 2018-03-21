#!/usr/bin/env python3 -O
#
# Example script to train gensim's doc2vec model on a wikified corpus.
#
# Words are treated as plain-text tokens and title tokens are "t:page_id:page_title"
#

from gensim.models.doc2vec import Doc2Vec
from .wikibrain_corpus import WikiBrainCorpus

import logging
import os
import os.path
import random
import sys


# As defined in https://arxiv.org/pdf/1607.05368.pdf
# We use these hyper-parameter values for WIKI (APNEWS): vector size = 300 (300), 
# window size = 15 (15), min count = 20 (10), sub-sampling threshold = 10−5 (10−5 ), 
# negative sample = 5, epoch = 20 (30). After removing low frequency words, the 
# vocabulary size is approximately 670K for WIKI and 300K for AP-NEW.
#
def train(args, corpus):
    alpha = 0.025
    min_alpha = 0.0001
    iters = args.iterations
    model = Doc2Vec(
        dm=0,
        vector_size=args.size,
        min_count=args.min_count,
        dbow_words=1,
        window=args.window,
        epochs=args.iterations,
        sample=1e-5,
        hs=0,
        negative=args.negative,
        alpha=alpha, min_alpha=alpha,
        workers=args.workers
    )
    sentences = list(corpus.get_sentences())
    model.build_vocab(sentences)
    for epoch in range(iters):
        logging.warn("BEGINNING ITERATION %d", epoch)
        random.shuffle(sentences)
        model.train(sentences, total_examples=len(sentences), epochs=1)

        # update alpha
        model.alpha -= (alpha - min_alpha) / iters
        model.alpha = max(model.alpha, min_alpha)
        model.min_alpha = model.alpha
    model.delete_temporary_training_data()
    return model

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build a fast text vector model from a wikibrain corpus.')
    parser.add_argument('--iterations',type=int, default=20, help='number of fast text iterations ')
    parser.add_argument('--size', type=int, default=300, help='size of vectors')
    parser.add_argument('--min_count', type=int, default=20, help='minimum word frequency')
    parser.add_argument('--window', type=int, default=15, help='window size')
    parser.add_argument('--negative', type=int, default=10, help='number of negative samples')
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count()), help='number of workers')
    parser.add_argument('--lower', action='store_true', default=False, help='whether to lowercase words in the corpus')
    parser.add_argument('--binary', action='store_true', default=False, help='whether to store the model as binary or text')
    parser.add_argument('--corpus', type=str, required=True, help='corpus directory')
    parser.add_argument('--output', type=str, required=True, help='vector file')
    parser.add_argument('--skip_entities', action='store_true', default=False, help='whether to compute vectors for articles as well as words')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s ' + args.corpus + ': %(message)s', level=logging.INFO)

    corpus = WikiBrainCorpus(args.corpus,
                             lower=args.lower,
                             entities=(not args.skip_entities),
                             min_freq=args.min_count)
    model = train(args, corpus)
    model.save_word2vec_format(args.output, binary=args.binary)


#!/usr/bin/env python3 -O
#
# Example script to train gensim's doc2vec model on a wikified corpus.
#
# Words are treated as plain-text tokens and title tokens are "t:page_id:page_title"
#

from gensim.models.doc2vec import Doc2Vec
from wmf_embed.train.wikibrain_corpus import WikiBrainCorpus

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
def train(corpus):
    alpha = 0.025
    min_alpha = 0.0001
    iters = 20
    model = Doc2Vec(
        dm=0,
        size=300,
        min_count=20,
        dbow_words=1,
        window=15,
        iter=iters,
        sample=1e-5,
        hs=0,
        negative=10,
        alpha=alpha, min_alpha=alpha,
        workers=min(8, os.cpu_count())
    )
    sentences = corpus.get_sentences()
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

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Build a fast text vector model from a wikibrain corpus.')
    parser.add_argument('--iterations',type=int, help='number of fast text iterations ')
    parser.add_argument('--size', type=int, help='size of vectors')
    parser.add_argument('--min_count', type=int, help='minimum word frequency')
    parser.add_argument('--window', type=int, help='window size')
    parser.add_argument('--negative', type=int, help='number of negative samples')
    parser.add_argument('--workers', type=int, help='number of workers')
    parser.add_argument('--binary', type=bool, help='whether the ')
    parser.add_argument('--lower', type=bool, default=False, help='whether to lowercase words in the corpus')
    parser.add_argument('--corpus', type=str, required=True, help='corpus directory')
    parser.add_argument('--output', type=str, required=True, help='vector file')

    args = parser.parse_args()

    params = FastTextParams()
    if args.iterations: params.iters = args.iterations
    if args.size: params.size = args.size
    if args.min_count: params.min_count = args.min_count
    if args.window: params.window = args.window
    if args.negative: params.negative = args.negative
    if args.workers: params.workers = args.workers

    logging.basicConfig(format='%(asctime)s ' + args.corpus + ': %(message)s', level=logging.INFO)

    corpus = WikiBrainCorpus(args.corpus, lower=args.lower, min_freq=args.min_count)
    vocab_size = write_fasttext(corpus, args.corpus + '/fasttext_corpus.txt')
    train_fasttext(args.corpus + '/fasttext_corpus.txt', args.output, vocab_size, params)

    if len(sys.argv) not in (4, 5):
        sys.stderr.write('usage: %s corpus_dir min_freq model_output_path [save_as_binary]\n' % sys.argv[0])
        sys.exit(1)

    binary = False
    if len(sys.argv) == 5:
        binary = str2bool(sys.argv[4])

    (corpus_dir, min_freq, output_path) = sys.argv[1:4]
    logging.basicConfig(format='%(asctime)s ' + corpus_dir + ': %(message)s', level=logging.INFO)
    min_freq = int(min_freq)
    corpus = WikiBrainCorpus(corpus_dir, min_freq=min_freq, lower=True)
    model = train(corpus)
    model.save_word2vec_format(output_path, binary=binary)


#!/usr/bin/env python3 -O
#
# Example script to train fasttext model on a wikified corpus.
#
# Words are treated as plain-text tokens and title tokens are "t:page_id:page_title"
#

from gensim.models.doc2vec import Doc2Vec
from urllib.request import urlretrieve

import logging
import math
import os
import os.path
import random
import subprocess
import sys
import zipfile


from .wikibrain_corpus import WikiBrainCorpus

def write_fasttext(corpus, path_out):
    vocab = set()
    with open(path_out, 'w', encoding='utf-8') as out:
        for sentence in corpus.get_sentences():
            words = sentence.words
            if words:
                tags = sentence.tags
                repeats_per_tag = int(math.ceil(len(words) / 7))
                for t in tags:
                    for i in range(repeats_per_tag):
                        words.insert(random.randint(0, len(words)), t)
                out.write(' '.join(words))
                out.write('\n')
                vocab.update(words)
                vocab.update(tags)
    return len(vocab)

def train_fasttext(path_corpus, path_vecs, vocab_size, args):
    if os.path.isdir('fastText-master/'):
        logging.info('fasttext already downloaded')
    else:
        url = 'https://github.com/facebookresearch/fastText/archive/master.zip'
        logging.info('downloading fast text from %s', url)
        urlretrieve(url, 'fasttext.zip')
        zip_ref = zipfile.ZipFile('fasttext.zip', 'r')
        zip_ref.extractall('./')
        zip_ref.close()

    if os.path.isfile('fastText-master/fasttext'):
        logging.info('fasttext already built')
    else:
        logging.info('building fasttext binary')
        subprocess.run(['make'], cwd='fastText-master', check=True)

    bucket_size = 2000000
    if vocab_size > 500000:
        bucket_size = 10000000

    subprocess.run(['./fastText-master/fasttext',
                    'cbow',
                    '-epoch', str(args.iterations),
                    '-neg', str(args.negative),
                    '-minCount', '10',
                    '-dim', str(args.size),
                    '-ws', str(args.window),
                    '-bucket', str(bucket_size),
                    '-input', path_corpus,
                    '-output', path_vecs
                    ],
                   check=True)

    os.unlink(path_vecs + '.bin')
    os.rename(path_vecs + '.vec', path_vecs)


if __name__ == '__main__':
    import argparse

    # Parameters based on https://arxiv.org/pdf/1802.06893.pdf
    parser = argparse.ArgumentParser(description='Build a fast text vector model from a wikibrain corpus.')
    parser.add_argument('--iterations',type=int, default=5, help='number of fast text iterations ')
    parser.add_argument('--size', type=int, default=300, help='size of vectors')
    parser.add_argument('--min_count', type=int, default=20, help='minimum word frequency')
    parser.add_argument('--window', type=int, default=5, help='window size')
    parser.add_argument('--negative', type=int, default=10, help='number of negative samples')
    parser.add_argument('--doc_lines', type=int, default=5, help='number of corpus lines to treat as pertaining to document')
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count()), help='number of workers')
    parser.add_argument('--lower', action='store_true', default=False, help='whether to lowercase words in the corpus')
    parser.add_argument('--corpus', type=str, required=True, help='corpus directory')
    parser.add_argument('--output', type=str, required=True, help='vector file')
    parser.add_argument('--skip_entities', action='store_true', default=False, help='whether to compute vectors for articles as well as words')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s ' + args.corpus + ': %(message)s', level=logging.INFO)

    corpus = WikiBrainCorpus(args.corpus,
                             lower=args.lower,
                             entities=(not args.skip_entities),
                             min_freq=args.min_count,
                             num_doc_lines=args.doc_lines)
    vocab_size = write_fasttext(corpus, args.corpus + '/fasttext_corpus.txt')
    train_fasttext(args.corpus + '/fasttext_corpus.txt', args.output, vocab_size, args)


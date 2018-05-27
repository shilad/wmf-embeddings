#!/usr/bin/env python3

import boto3
import botocore
import logging

import itertools

LANGS = 'de	en	es	fa	it	simple' .split()

INFO = [
    ['vectors.fasttext.lower.txt.bz2', 'w2v3_plain', 'fasttext', 'lower', -1 ],
    ['vectors.fasttext.mixed.txt.bz2', 'w2v3_plain', 'fasttext', 'mixed', -1 ],
    ['vectors.word2vec.lower.txt.bz2', 'w2v3_plain', 'doc2vec', 'lower', -1 ],
    ['vectors.word2vec.mixed.txt.bz2', 'w2v3_plain', 'doc2vec', 'mixed', -1 ],
    ['vectors.doc2vec.all_doc_lines.lower.txt.bz2', 'w2v3', 'doc2vec', 'lower', 10000 ],
    ['vectors.doc2vec.all_doc_lines.mixed.txt.bz2', 'w2v3', 'doc2vec', 'mixed', 10000 ],
    ['vectors.doc2vec.lower.txt.bz2', 'w2v3', 'doc2vec', 'lower', 5 ],
    ['vectors.doc2vec.mixed.txt.bz2', 'w2v3', 'doc2vec', 'mixed', 5 ],
    ['vectors.doc2vec.no_doc_lines.lower.txt.bz2', 'w2v3', 'doc2vec', 'lower', 0 ],
    ['vectors.doc2vec.no_doc_lines.txt.bz2', 'w2v3', 'doc2vec', 'mixed', 0 ],
    ['vectors.fasttext.all_doc_lines.lower.txt.bz2', 'w2v3', 'fasttext', 'lower', 10000 ],
    ['vectors.fasttext.all_doc_lines.mixed.txt.bz2', 'w2v3', 'fasttext', 'mixed', 10000 ],
    ['vectors.fasttext.lower.no_doc_lines.txt.bz2', 'w2v3', 'fasttext', 'lower', 0 ],
    ['vectors.fasttext.lower.txt.bz2', 'w2v3', 'fasttext', 'lower', 5 ],
    ['vectors.fasttext.mixed.txt.bz2', 'w2v3', 'fasttext', 'mixed', 5 ],
    ['vectors.fasttext.no_doc_lines.txt.bz2', 'w2v3', 'fasttext', 'mixed', 0 ],
    ['vectors.fasttext.noentities.lower.txt.bz2', 'w2v3', 'fasttext', 'lower', -1 ],
    ['vectors.fasttext.noentities.mixed.txt.bz2', 'w2v3', 'fasttext', 'mixed', -1 ],
    ['vectors.word2vec.noentities.lower.txt.bz2', 'w2v3', 'doc2vec', 'lower', -1 ],
    ['vectors.word2vec.noentities.mixed.txt.bz2', 'w2v3', 'doc2vec', 'mixed', -1 ]
]

S3_BUCKET = 'wikibrain'

class ResultInfo:
    def __init__(self, name, corpus, bucket, path, lang, alg, case, num_doc_lines):
        self.name = name
        self.corpus = corpus
        self.bucket = bucket
        self.path = path
        self.lang = lang
        self.alg = alg
        self.case = case
        self.num_doc_lines = num_doc_lines
        self.sr_results = {
                    'found' : 0,
                    'not-found' : 0,
                    'accuracy': 0
                }
        self.analogy_results = {
                    'found' : 0,
                    'not-found' : 0,
                    'accuracy': 0
                }

    def sr_coverage(self):
        n = self.sr_results['found'] + self.sr_results['not-found']
        return 0.0 if n == 0 else (1.0 * self.sr_results['found'] / n)

    def analogy_coverage(self):
        n = self.analogy_results['found'] + self.analogy_results['not-found']
        return 0.0 if n == 0 else (1.0 * self.analogy_results['found'] / n)

def main():
    infos = []
    for name, corpus, alg, case, num_doc_lines in INFO:
        for lang in LANGS:
            if name.endswith('.bz2'):
                name = name[:-4]
            path = '/'.join([corpus, lang, 'eval.mono.' + name + '.txt'])
            infos.append(ResultInfo(name, corpus,
                                    S3_BUCKET, path,
                                    lang, alg, case, num_doc_lines))

    s3 = boto3.resource('s3')
    for i in infos:
        obj = s3.Object(i.bucket, i.path)
        try:
            text = obj.get()['Body'].read().decode('utf-8')
            parse_results(i, text)
        except Exception as e:
            if 'NoSuchKey' in str(e):
                logging.warning('Couldn\'t find %s in %s', i.path, i.bucket)
            else:
                raise

    sr_langs = sorted({i.lang for i in infos if i.sr_results['found'] > 0})
    analogy_langs = sorted({i.lang for i in infos if i.analogy_results['found'] > 0})

    f = open('./results.tsv', 'w')
    tokens = ['name', 'corpus', 'alg', 'case', 'doc_lines']
    for lang in sr_langs + analogy_langs:
        tokens.extend([lang, lang + '-m'])
    f.write('\t'.join([str(t) for t in tokens]) + '\n')

    for _, lang_infos in itertools.groupby(infos, lambda i: i.name + i.corpus):
        by_lang = {i.lang : i for i in lang_infos}
        li = list(by_lang.values())[0]
        tokens = [li.name, li.corpus, li.alg, li.case, li.num_doc_lines]
        for l in sr_langs:
            r = by_lang.get(l)
            if r:
                tokens.extend([r.sr_results['accuracy'], r.sr_coverage()])
            else:
                tokens.extend([0, 0.0])

        for l in analogy_langs:
            r = by_lang.get(l)
            if r and r.sr_results['found'] > 0:
                tokens.extend([r.analogy_results['accuracy'], r.analogy_coverage()])
            else:
                tokens.extend([0, 0.0])

        f.write('\t'.join([str(t) for t in tokens]) + '\n')


    f.close()


def parse_results(info, res):
    for line in res.split('\n'):
        line = line.strip().lower()
        if line.startswith('info:root:'):
            line = line[10:].strip().split()
            if len(line) >= 4:
                if line[0] == 'dataset':
                    res = info.sr_results
                elif line[0] == 'category':
                    res = info.analogy_results
                elif line[0] == 'overall':
                    res['found'] = int(line[1])
                    res['not-found'] = int(line[2])
                    res['accuracy'] = float(line[3])


if __name__ == '__main__':
    main()


#!/usr/bin/python -O

from collections import defaultdict
import logging
import os.path

from utils import Titler, LangEmbedding

def main(dir):
    titler = Titler(os.path.join(dir, 'titles.csv'))
    embeddings = []

    for lang in os.listdir(dir):
        p = os.path.join(dir, lang)
        if os.path.isdir(p) and (lang == 'nav' or lang.endswith('wiki')):
            emb = LangEmbedding(lang, p, titler, aligned=True)
            emb.build_fast_knn()
            embeddings.append(emb)


    synonyms = defaultdict(list)
    for emb in embeddings:
        for id in emb.ids:
            if not id[:2] in ('c:', 'p:') and ':' in id:
                proj, word = id.split(':', 1)
                synonyms[word].append(proj)

    with open(os.path.join(dir, 'word_neighbors.tsv'), 'w', encoding='utf-8') as f:
        for word, projs in synonyms.items():
            if len(projs) >= 2:
                f.write('\t'.join([p + ':' + word for p in projs]))
                f.write('\n')

    return

    for emb in embeddings:
        p = os.path.join(dir, emb.lang, 'neighbors.tsv')
        with open(p, 'w', encoding='utf-8') as f:
            for i, id in enumerate(emb.ids):
                if i % 10000 == 0:
                    logging.info('generating neighbors for id %d of %d in %s', i, len(emb.ids), emb.lang)
                neighbors = emb.neighbors(id, n=10, include_distances=True)
                neighbor_ids = [id2 for (id2, dist) in neighbors if id != id2 and dist < 1.0]
                f.write('\t'.join([id] + neighbor_ids) + '\n')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main('./output')
#!/usr/bin/python3
import bz2
import gzip
import logging
import random

import os.path
import re

import pandas as pd
import numpy as np

class AlignmentParams:
    def __init__(self, path_in, path_out):
        self.path_in = path_in
        self.path_out = path_out
        self.path_sitelinks = os.path.join(path_in, 'sitelinks.csv')

def main(params):
    sitelinks = pd.read_csv(params.path_sitelinks)
    logging.info('read sitelinks %s with %d entries', params.path_sitelinks, len(sitelinks))

    for fname in os.listdir(params.path_in):
        parts = fname.split('.')
        if len(parts) >= 3 and parts[0] == 'vectors' and parts[2] == 'txt':
            wiki = parts[1]
            pin = os.path.join(params.path_in, fname)
            dout = os.path.join(params.path_out, wiki)
            if wiki == 'nav':
                translate_nav_vectors(pin, dout)
            else:
                translate_content_vectors(sitelinks, wiki, pin, dout)


def wiki_to_project(wiki):
    if wiki.endswith('wiki'):
        wiki = wiki[:-4]
    return wiki + '.wikipedia'


is_num = re.compile('^\d+$').match

def translate_nav_vectors(path_in, dir_out):
    ids = []
    vectors = []
    for id, vector in read_mikolov_txt(path_in):
        if is_num(id):
            id = 'c:' + id
        ids.append(id)
        vectors.append(vector)

    if not os.path.isdir(dir_out): os.makedirs(dir_out, exist_ok=True)
    np.save(os.path.join(dir_out, 'vectors.np'), np.array(vectors))
    with open(os.path.join(dir_out, 'ids.txt'), 'w', encoding='utf-8') as f:
        for line in ids:
            f.write(line + '\n')

def translate_content_vectors(site_links, wiki, path_in, dir_out):
    for_wiki = site_links[site_links['sitelinks.wiki_db'] == wiki]
    page_ids = for_wiki['sitelinks.page_id']
    entities = for_wiki['sitelinks.entity']
    id_to_entity = dict(zip(page_ids, entities))
    project = wiki_to_project(wiki)

    result_ids = []
    result_vectors = []
    rows = 0

    for id, vector in read_mikolov_txt(path_in):
        rows += 1
        if id.startswith('t:'):
            page_id = int(id.split(':')[1])
            if page_id in id_to_entity:
                id = 'c:' + id_to_entity[page_id][1:]  # drop the 'Q'
            else:
                id = 'p:' + str(page_id)
        else:
            id = project + ':' + id
        result_ids.append(id)
        result_vectors.append(vector)

    if not os.path.isdir(dir_out): os.makedirs(dir_out, exist_ok=True)
    np.save(os.path.join(dir_out, 'vectors.np'), np.array(result_vectors))
    with open(os.path.join(dir_out, 'ids.txt'), 'w', encoding='utf-8') as f:
        for line in result_ids:
            f.write(line + '\n')

def read_mikolov_txt(path):
    with open_text_file(path) as f:
        header = f.readline()
        for i, line in enumerate(f):
            if i % 100000 == 0:
                logging.info("reading line %d of %s" % (i, path))
            line = line.rstrip('\n ')
            tokens = line.split(' ')
            id = tokens[0]
            vector = np.array([float(x) for x in tokens[1:]])
            yield(id, vector)


def open_text_file(path):
    if path.lower().endswith('bz2') or path.lower().endswith('bz'):
        return bz2.open(path, 'rt', encoding='utf-8', errors="backslashreplace")
    elif path.lower().endswith('gz'):
        return gzip.open(path, 'rt', encoding='utf-8', errors="backslashreplace")
    else:
        return open(path, 'r', encoding='utf-8', errors="backslashreplace")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(AlignmentParams('./input', './output'))
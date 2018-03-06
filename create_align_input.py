#!/usr/bin/python3
import bz2
import gzip
import logging
import random

import os.path
import re

import pandas as pd
import numpy as np

from utils import NP_FLOAT


class AlignmentParams:
    def __init__(self, path_in, path_out):
        self.path_in = path_in
        self.path_out = path_out
        self.title_lang = 'simple.wikipedia'
        self.path_sitelinks = os.path.join(path_in, 'sitelinks.csv')

def main(params):
    sitelinks = pd.read_csv(params.path_sitelinks)
    logging.info('read sitelinks %s with %d entries', params.path_sitelinks, len(sitelinks))

    ids = set()
    for fname in os.listdir(params.path_in):
        parts = fname.split('.')
        if len(parts) >= 3 and parts[0] == 'vectors' and parts[2] == 'txt':
            wiki = parts[1]
            pin = os.path.join(params.path_in, fname)
            dout = os.path.join(params.path_out, wiki)
            if wiki == 'nav':
                new_ids = translate_nav_vectors(pin, dout)
            else:
                new_ids = translate_content_vectors(sitelinks, wiki, pin, dout)
            ids.update(new_ids)

    write_titles(sitelinks, ids, os.path.join(params.path_out, 'titles.csv'), params.title_lang)

def write_titles(sitelinks, ids, out_path, title_proj):
    # Add placeholders so we know what we care about.
    needed = set()
    for id in ids:
        if id.startswith('c:'):
            needed.add(('concept', id[2:]))
        else:
            parts = id.split(':')
            if len(parts) >= 3 and parts[1] == 'page':
                needed.add((parts[0], parts[2]))

    rows = []

    for index, row in sitelinks.iterrows():
        proj = wiki_to_project(row['sitelinks.wiki_db'])
        concept = row['sitelinks.entity'][1:]   # drop 'Q'
        page_id = str(row['sitelinks.page_id'])
        title = row['sitelinks.effective_page_title']

        if proj == title_proj and ('concept', concept) in needed:
            rows.append(('concept', concept, title))
            needed.remove(('concept', concept))
        if (proj, page_id) in needed:
            rows.append((proj, page_id, title))
            needed.remove((proj, page_id))
    df = pd.DataFrame(rows, columns=('project', 'page_id', 'title'))
    df.to_csv(out_path, index=False)

def wiki_to_project(wiki):
    if wiki.endswith('wiki'):
        wiki = wiki[:-4]
    return wiki + '.wikipedia'

def project_to_wiki(project):
    return project.split('.')[0] + 'wiki'


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
    np.save(os.path.join(dir_out, 'vectors'), np.array(vectors, dtype=NP_FLOAT))
    with open(os.path.join(dir_out, 'ids.txt'), 'w', encoding='utf-8') as f:
        for line in ids:
            f.write(line + '\n')
    return ids

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
                id = project + ':page:' + str(page_id)
        else:
            id = project + ':' + id
        result_ids.append(id)
        result_vectors.append(vector)

    if not os.path.isdir(dir_out): os.makedirs(dir_out, exist_ok=True)
    np.save(os.path.join(dir_out, 'vectors'), np.array(result_vectors, dtype=NP_FLOAT))
    with open(os.path.join(dir_out, 'ids.txt'), 'w', encoding='utf-8') as f:
        for line in result_ids:
            f.write(line + '\n')
    return result_ids

def read_mikolov_txt(path):
    with open_text_file(path) as f:
        header = f.readline()
        for i, line in enumerate(f):
            if i % 100000 == 0:
                logging.info("reading line %d of %s" % (i, path))
            line = line.rstrip('\n ')
            tokens = line.split(' ')
            id = tokens[0]
            vector = np.array([float(x) for x in tokens[1:]], dtype=NP_FLOAT)
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
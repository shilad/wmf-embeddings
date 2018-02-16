#!/usr/bin/env python3

import json
import logging
import os.path
import pandas as pd
import numpy as np
import bz2
import gzip

import re


def main(project, path_sessions, path_nav_vectors, path_content_vectors, path_sitelinks, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    sitelinks = pd.read_csv(path_sitelinks)
    ids = translate_content_vectors(path_content_vectors, out_path, sitelinks, project)
    translate_sessions(path_sessions, project, ids, out_path)
    translate_nav_vectors(path_nav_vectors, ids, out_path, project)


def translate_content_vectors(path_in, path_out, sitelinks, lang):
    pairs = zip(sitelinks.page_id, sitelinks.entity)
    page_to_entity = {page: int(entity[1:]) for (page, entity) in pairs if entity[:1] == 'Q'}
    ids = []
    matrix = []
    for (id, vector) in read_mikolov_txt(path_in):
        if id.startswith('t:'):
            page_id = int(id.split(':')[1])
            if page_id in page_to_entity:
                ids.append(str(page_to_entity[page_id]))
                matrix.append(vector)
        else:
            ids.append(lang + ':' + id)
            matrix.append(vector)

    # Write out all the ids
    with open(path_out + '/external_ids.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(ids) + '\n')

    # Write out the ids for content vectors (which are all of them!)
    with open(path_out + '/content_ids.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join([str(i) for i in range(len(ids))]) + '\n')

    # Write out the content vectors
    np.save(path_out + '/content_vectors.npy', np.array(matrix))

    return ids

def translate_sessions(path_in, project, ids, path_out):
    id_to_index = { id : i for (i, id) in enumerate(ids) }
    sessions = pd.read_csv(path_in, encoding='utf-8')
    result = []
    for view_str, search_str in zip(sessions['sessions.views'], sessions['sessions.searches']):
        views = json.loads(view_str)
        searches = json.loads(search_str)
        session = [(str(v['wikidata_id']), v['seconds_after_start']) for v in views] + \
                  [(project + ':' + s['query'], s['seconds_after_start']) for s in searches]
        session.sort(key=lambda x: x[1])
        indexes = [str(id_to_index[id]) for id, tstamp in session if id in id_to_index]
        result.append(' '.join(indexes))
    with open(path_out + '/sessions.txt', 'w') as f:
        f.write('\n'.join(result) + '\n')




def translate_nav_vectors(path_in, ids, path_out, lang):
    id_to_index = { id : i for (i, id) in enumerate(ids) }

    ids = []
    matrix = []
    for id, vector in read_mikolov_txt(path_in):
        if id in id_to_index:
            ids.append(str(id_to_index[id]))
            matrix.append(vector)

    # Write out the ids
    with open(path_out + '/nav_ids.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(ids) + '\n')

    # Write out the content vectors
    np.save(path_out + '/nav_vectors.npy', np.array(matrix))

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
    main('simple.wikipedia',
         'sessions.simple.csv',
         'vectors.txt.bz',
         'vectors_200_w101.txt',
         'simple_sitelinks.csv',
         'dataset')
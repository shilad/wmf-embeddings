#!/usr/bin/python3
#
# Creates input matrices for different languages suitable for numpy
#
#


import logging
import multiprocessing
import os.path
import re
import sys

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from utils import NP_FLOAT


def main(path_in, path_out):
    read_sitelinks(path_in)

    wikis = set()
    ids = set()

    w2v_paths = [os.path.join(path_in, p) for p in os.listdir(path_in)]
    w2v_paths = [(p, path_out) for p in w2v_paths if is_word2vec_model(p) ]
    with multiprocessing.Pool(4) as pool:
        for wiki, wiki_ids in pool.imap_unordered(process_w2v, w2v_paths):
            wikis.add(wiki)
            if wiki != 'nav':
                ids.update(wiki_ids)

    if len(wikis) == 1:
        logging.error('only found wikis %s in %s' % (wikis, path_in))
        sys.exit(1)

    if 'enwiki' in wikis:
        title_lang = 'enwiki'
    elif 'simplewiki' in wikis:
        title_lang = 'simplewiki'
    else:
        title_lang = 'enwiki'

    write_titles(sitelinks, ids, os.path.join(path_out, 'titles.csv'), title_lang)


def read_sitelinks(path_in):
    global sitelinks

    path1 = os.path.join(path_in, 'sitelinks.csv')
    if os.path.isfile(path1):
        logging.info('reading sitelinks from %s' % path1)
        sitelinks = pd.read_csv(path1, encoding='utf-8', error_bad_lines=False)
        logging.info('read sitelinks %s with %d entries', path1, len(sitelinks))
        return

    path2 = os.path.join(path_in, 'sitelinks.tsv.bz2')
    if os.path.isfile(path2):
        logging.info('reading sitelinks from %s' % path2)
        sitelinks = pd.read_csv(path2, delimiter='\t', encoding='utf-8', error_bad_lines=False)
        logging.info('read sitelinks %s with %d entries', path2, len(sitelinks))
        return

    sys.stderr.write('couldnt find sitelinks in %s or %s!' % (path1, path2))
    sys.exit(1)

def is_word2vec_model(path):
    """
    Checks to see if the path corresponds to a word2vec file.
    Looks for a filename such as "vectors.enwiki.bin" or "vectors.enwiki.txt.bz2"
    """
    if not os.path.isfile(path):
        return False
    parts = os.path.basename(path).split('.')
    if len(parts) not in (3, 4):
        return False
    elif len(parts) == 4 and parts[-1].lower() not in ('bz', 'bz2', 'gz'):
        return False
    return parts[0] == 'vectors' and parts[2] in ('bin', 'txt')

def process_w2v(paths):
    global sitelinks

    (path_in, path_out) = paths

    parts = os.path.basename(path_in).split('.')
    wiki = parts[1]
    dir_out = os.path.join(path_out, wiki)
    logging.info('processing word2vec model %s (wiki = %s)', path_in, wiki)

    if not os.path.isdir(dir_out):
        os.makedirs(dir_out, exist_ok=True)

    if wiki == 'nav':
        new_ids = translate_nav_vectors(path_in, dir_out)
    else:
        new_ids = translate_content_vectors(sitelinks, wiki, path_in, dir_out)

    return (wiki, new_ids)


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
        try:
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
        except:
            logging.exception('Row %s failed', str(row))

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
    for id, vector in read_word2vec(path_in):
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

    for id, vector in read_word2vec(path_in):
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

def read_word2vec(path):
    # vectors.enwiki.txt.bz2, vectors.enwiki.bin etc.
    parts = os.path.basename(path).split('.')
    assert(parts[2] in ('bin', 'txt'))

    model = KeyedVectors.load_word2vec_format(path,
                                              datatype=NP_FLOAT,
                                              encoding='utf-8',
                                              unicode_errors='ignore',
                                              binary=(parts[2] == 'bin'))

    for i, id in enumerate(model.index2word):
        yield (id, model.vectors[i,:])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 3:
        sys.stderr.write('usage: %s path_in path_out' % (sys.argv[0]))
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

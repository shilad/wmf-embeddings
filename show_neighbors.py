#!/usr/bin/env python3
#
# A script that shows neighbors for user-specified queries.
#
import io
import re

import requests
import sys

from embedding import Embedding

def is_number(s):
    return all([c in '0123456789' for c in s])

def entities_to_titles(ids):
    if len(ids) > 50:
        return entities_to_titles(ids[:50]) + entities_to_titles(ids[50:])
    wd_ids = '|'.join('Q' + i for i in ids)
    url = "https://www.wikidata.org/w/api.php" \
          "?action=wbgetentities" \
          "&ids=%s" \
          "&languages=en" \
          "&format=json" \
          "&props=labels|descriptions" % (wd_ids)

    js = requests.get(url).json()

    results = []
    for id in ('Q' + i for i in ids):
        title = 'Unknown'
        if id in js.get('entities', {}):
            info = js['entities'][id]
            if 'labels' in info and 'en' in info['labels']:
                title = info['labels']['en']['value']
            elif 'descriptions' in info and 'en' in info['descriptions']:
                title = info['descriptions']['en']['value']
        results.append('%s: %s' % (id, title))

    return results

def describe_ids(ids):
    entity_ids = []
    for i in ids:
        if is_number(i) or i[:1] == 'Q' and is_number(i[1:]):
            entity_ids.append(i)
    wd_id_to_title = dict(zip(entity_ids, entities_to_titles(entity_ids)))
    descriptions = []
    for i in ids:
        descriptions.append(wd_id_to_title.get(i, i))
    return descriptions

def query_to_id(query):
    word = query.strip()
    if all([c in '0123456789' for c in word]):  # Numeric Wikidata id?
        key = word
    elif word[0] == 'Q' and all([c in '0123456789' for c in word[1:]]): # Q-prefixed Wikidata id?
        key = word[1:]
    elif 'wikipedia:' in word: # project-specific query
        key = re.sub('//s+', '_', word)
    else:   # textual query.
        #key = 'en.wikipedia:' + re.sub('//s+', '_', word)
        key = re.sub('//s+', '_', word)
    return key

inp = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
emb = Embedding(sys.argv[1])
while True:
    sys.stdout.write("Enter search string, wikidata id, or \"quit\": ")
    sys.stdout.flush()
    line = inp.readline()
    if line.strip().lower() == 'quit':
        break
    id = query_to_id(line.strip())
    if not emb.has_id(id):
        print('id %s not found in index.' % (repr(id), ))
        continue
    neighbor_ids = emb.neighbors(id, 100)
    print('\nresults for query: %s' % (repr(describe_ids([id])[0]), ))
    for i, desc in enumerate(describe_ids(neighbor_ids)):
        print('%d. %s' % (i+1, desc))

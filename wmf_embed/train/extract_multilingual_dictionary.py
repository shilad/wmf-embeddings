#!/usr/bin/env python3 -O
#
# Author: Shilad Sen
#
# Generates a multilingual dictionary of synonyms from the wikidata dump.
# Only unique (unambiguous) synonyms are kept. All words are converted to
# lowercase and spaces are replaced with tabs.
#
# Output file is one tab-separated synonym set per line. For example:
#
# en.wikipedia:life   fr.wikipedia:vie    eo.wikipedia:vivo   it.wikipedia:vita   ...
#

import json
import logging
import os
import pybloomfilter
from tempfile import NamedTemporaryFile, TemporaryFile

import smart_open
import sys
import traceback

NUM_LABELS = 200 * 10 ** 6  # 200M labels

def main(input_path, output_path):
    all = pybloomfilter.BloomFilter(NUM_LABELS, 0.0001)
    dups = pybloomfilter.BloomFilter(NUM_LABELS, 0.0001)
    logging.info('creating bloom filter of %dMBs', all.num_bits / 8 // (1024 * 1024))
    num_distinct = 0
    num_dups = 0

    input = smart_open.smart_open(input_path, 'rb')
    tmp_output = TemporaryFile(mode='w+', encoding='utf-8')
    for i, line in enumerate(input):
        if i % 10000 == 0:
            logging.info('processing line %d. %d unique, %d dups', i, num_distinct - num_dups, num_dups)
        try:
            line = line.decode('utf-8').strip()
            if len(line) <= 2: continue # opening or closing
            if line[-1] == ',': line = line[:-1]
            data = json.loads(line)
            result = []
            for labels in data['labels'].values():
                lang = labels['language']
                word = lang + '.wikipedia:' + labels['value'].lower().replace(' ', '_')
                if word in all:
                    dups.add(word)
                    num_dups += 1
                else:
                    result.append(word)
                    all.add(word)
                    num_distinct += 1
            tmp_output.write('\t'.join(result))
            tmp_output.write('\n')
        except:
            sys.stderr.write('error while processing line: %s' % (line))
            traceback.print_exc()

    logging.info('found %d unique, %d dups', num_distinct - num_dups, num_dups)
    tmp_output.seek(0)

    with open(output_path, 'w', encoding='utf-8') as output:
        for line in tmp_output:
            words = [
                w
                for w in line.strip().split('\t')
                if w not in dups
            ]
            if len(words) > 1:
                output.write('\t'.join(words))
                output.write('\n')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('usage: %s path_wikidata_json output_path', sys.argv[0])
        sys.exit(1)
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1], sys.argv[2])
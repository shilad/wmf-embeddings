#!/usr/bin/python3 -O
#
# A simple script to extract sitelinks from the wikidata dump.
#
# Usage: python3 extract_wikidata_links.py latest.all.json.bz2 sitelinks.tsv
#
# The input file must be a Wikidata json dump, either bzipped or decompressed.
# The output file is a tsv with columns 1) entity id, 2) wiki_db, and 3) title
#
# If an error occurs, it is logged and processing continues.
#


import bz2
import sys
import json
import traceback

if len(sys.argv) != 3:
    sys.stderr.write('Usage: %s path/wikidata_json.bz2 path/output.tsv\n', sys.argv[0])
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]


if input_path.lower().endswith('bz2'):
    input = bz2.open(input_path, 'rt', encoding='UTF-8')
else:
    input = open(input_path, encoding='UTF-8')

output = open(output_path, 'w', encoding='UTF-8')

for i, line in enumerate(input):
    if i % 10000 == 0:
        sys.stderr.write('processing line %d\n' % i)
    try:
        line = line.strip()
        if line[-1:] == ',':
            line = line[:-1]
        if line and line[:1] == '{' and line[-1:] == '}':
            js = json.loads(line)
            id = js['id']
            for siteinfo in js.get('sitelinks', {}).values():
                output.write(id + '\t' + siteinfo['site'] + '\t' + siteinfo['title'] + '\n')
    except:
        sys.stdout.write('failure in line %d (%s...)\n' % (i+1, line[:-40]))
        traceback.print_exc()

output.close()


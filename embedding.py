#!/usr/bin/env python3
#
# Reads a word2vec model in non-binary format and builds an annoy index around
# it for fast nearest neighbor queries.
#

import annoy
import json
import os
import os.path
import pickle
import re
import urllib.request

import sys


class Embedding:
    def __init__(self, path):
        self.model_path = path
        self.annoy_path = path + '.ann'
        self.id_path = path + '.ids'
        self.mtime_path = path + '.mtime'
        self.index = None           # annoy accelerated KNN index
        self.idToIndex = None       # External string id to internal numeric annoy
        self.indexToId = None       # Internal numeric annoy id to sxternal string id

        if not self.is_built():
            self.build()

        self.load()

    def is_built(self):
        all_paths = (self.model_path, self.annoy_path, self.id_path, self.mtime_path)
        if not all(os.path.exists(p) for p in all_paths):
            return False
        actual_mtime = str(os.path.getmtime(self.model_path))
        built_mtime = str(open(self.mtime_path).read().strip())
        return actual_mtime == built_mtime

    def load(self):
        sys.stderr.write('loading ids...\n')
        with open(self.id_path, 'rb') as f:
            (self.dims, self.idToIndex, self.indexToId) = pickle.load(f)
        sys.stderr.write('loading index...\n')
        self.index = annoy.AnnoyIndex(self.dims, metric='angular')
        self.index.load(self.annoy_path)
        assert(len(self.idToIndex) == len(self.indexToId))
        sys.stderr.write('loaded embedding for %d ids\n' % len(self.idToIndex))

    def build(self):
        model_file = open(self.model_path, 'rb')
        header = model_file.readline()
        rows, dims = map(int, header.split())

        # Annoy expects dense numeric ids, so we need to create a mapping back and forth.
        # We call the sparse string index an "id" and the dense numeric number an "index"
        idToIndex = {}
        indexToId = []

        # Build the approximate-nearest-neighbor index
        index = annoy.AnnoyIndex(dims, metric='angular')
        for i, bytes in enumerate(model_file):
            if i % 10000 == 0:
                sys.stderr.write('reading %s (line %d of %d)\n' % (sys.argv[1], i, rows))
            try:
                line = bytes.decode('utf-8')
            except:
                sys.stderr.write('couldn\'t decode bytes: %s' % (repr(bytes[:100])))
                continue

            line = line.rstrip(' \n')
            tokens = line.split(' ')
            if len(tokens) != dims + 1:
                sys.stderr.write('invalid line: %s\n' % (repr(line), ))
                continue
            i = len(idToIndex)
            id = tokens[0]
            idToIndex[id] = i
            indexToId.append(id)
            vec = [float(x) for x in tokens[1:]]
            index.add_item(i, vec)

        index.build(10)
        index.save(self.annoy_path)
        del(index)

        # write ids
        with open(self.id_path, 'wb') as f:
            pickle.dump((dims, idToIndex, indexToId), f)

        # write mtime
        with open(self.mtime_path, 'w') as f:
            f.write(str(os.path.getmtime(self.model_path)))

    def has_id(self, id):
        return id in self.idToIndex

    def neighbors(self, id, n=100):
        if id not in self.idToIndex:
            return []
        res = self.index.get_nns_by_item(self.idToIndex[id], n)
        return [self.indexToId[i] for i in res]

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.stderr.write("usage: %s [path-vector-model.txt]\n"  % sys.argv[0])
        sys.exit(1)

    e = Embedding(sys.argv[1])
    print(e.neighbors("es.wikipedia:gato"))

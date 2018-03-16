#!/usr/bin/env python3
#
# A script that evaluates the accuracy of a word2vec embedding at predicting pageviews within a session
# The vector model is expected to be in non-binary word2vec format as output by Mikolov's word2vec.c.
#
# The script calculates "hit-rates" for session prediction. For session prediction,
#

import sys

from wmf_embed.model import embedding

if len(sys.argv) != 4:
    sys.stderr.write("usage: %s [path-vector-model.txt] [path-test-data.txt] num-sessions"  % sys.argv[0])
    sys.exit(1)

emb = embedding.Embedding(sys.argv[1])

numTestSessions = int(sys.argv[3])
hitRanks = []   # Ranks of first hit, or -1 if one doesn't exist.
numTokens = 0

for (i, line) in enumerate(open(sys.argv[2])):
    if i >= numTestSessions:
        break

    if i % 100 == 0:
        sys.stderr.write('Evaluating testing session %d of %d.\n' % (i, numTestSessions))

    tokens = line.strip().split(' ')
    numTokens += len(tokens)

    indexes = [idToIndex[id] for id in tokens if id in idToIndex ]

    for j in range(len(indexes) - 1):
        neighbors = index.get_nns_by_item(indexes[j], 100)
        neighbors = [n for n in neighbors if n != indexes[j]]   # remove the id itself from the list

        hits = set(indexes[j:])
        rank = -1   # -1 indicates not found
        for (r, n) in enumerate(neighbors):
            if n in hits:
                rank = r
                break
        hitRanks.append(rank + 1)   # rank should start at 1, not 0


print('coverage: %.3f (%d of %d)' % (1.0 * len(hitRanks) / numTokens,
                                     len(hitRanks),
                                     numTokens))

for rank in (1, 5, 20, 100):
    nHits = len([r for r in hitRanks if r > 0 and r <= rank])
    print('hit rate within top-%d: %.3f (%d of %d)' % (rank, nHits / len(hitRanks),
                                                       nHits,
                                                       len(hitRanks)))






#!/bin/bash
#
# Script to build all aligned word2vec models.
#

set -e
set -x

langs="ar az bg ca cs da de en eo es et eu fa fi fr gl he hi hr hu id it ja kk ko lt ms nl nn no pl pt ro ru simple sk sl sr sv tr uk vi vo war zh"

# Download all necessary files
#[ -f ./data/input ] || mkdir -p ./data/input

#aws s3 cp s3://wikibrain/w2v2/sitelinks.tsv.bz2 ./data/input/
#aws s3 cp s3://wikibrain/w2v2/vectors.nav.txt.bz2 ./data/input/
#for lang in $langs; do
#     aws s3 cp s3://wikibrain/w2v2/$lang/vectors.$lang.bin ./data/input/vectors.${lang}wiki.bin
#done

#python3 ./create_align_input.py ./data/input ./data/output
#python3 ./make_neighbor_graphs.py ./data/output
echo $langs | tr ' ' '\n' | parallel --no-notice -j 10 python3 ./hybrid_vectors.py ./data/output {}wiki

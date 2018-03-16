#!/bin/bash
#
# Rebuilds word2vec models for a particular language
# usage: build-fasttext.sh [model-file-name] [lang1,lang2,..]
#        build-fasttext.sh [model-file-name]
#        build-fasttext.sh
#

set -e
set -x

langs="en,de,simple,ar,az,bg,ca,cs,da,eo,es,et,eu,fa,fi,fr,gl,he,hi,hr,hu,id,it,ja,kk,ko,lt,ms,nl,nn,no,pl,pt,ro,ru,sk,sl,sr,sv,tr,uk,vi,vo,war,zh"
name=vectors.fasttext.txt
script_args=""

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --languages)
        langs="$2"
        shift # past argument
        shift # past value
        ;;
        --name)
        name="$2"
        shift # past argument
        shift # past value
        ;;
        --)
        shift # past argument
        script_args=$@
        break
        ;;
        *)    # unknown option
        echo "
        Usage: [--languages en,de,...]
               [--name vectors.fasttext.txt]
               [-- extra arguments to script]
       " >&2
       exit 1
        ;;
    esac
done


echo building word2vec model $name for languages $langs with extra args of $script_args >&2


function do_lang() {
    name=$1
    lang=$2
    shift
    shift
    extra_args=$@
    mkdir -p ./w2v/$lang
    aws s3 cp s3://wikibrain/w2v2/$lang/corpus.txt.bz2 ./w2v/$lang/
    aws s3 cp s3://wikibrain/w2v2/$lang/dictionary.txt.bz2 ./w2v/$lang/
    python36 -m wmf_embed.train.train_fasttext ./w2v/$lang/ 300 20 ./w2v/$lang/$name $extra_args 2>&1 | tee ./w2v/$lang/${name}.log
    pbzip2 ./w2v/$lang/$name
    aws s3 cp ./w2v/$lang/${name}.bz2 s3://wikibrain/w2v2/$lang/
    rm -rf ./w2v/$lang/
}

export -f do_lang

echo $langs | tr ',' '\n' | parallel -j 6 --line-buffer do_lang $name '{}'

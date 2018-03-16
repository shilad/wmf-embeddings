#!/bin/bash
#
# Rebuilds fasttext models models for a particular language
#
#  Usage: build_fasttext.sh                     \
#         [--languages en,de,...]               \
#         [--name vectors.fasttext.txt]         \
#         [-- extra arguments to script]
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
    path_vecs=./vecs/${name}/${lang}
    mkdir -p ${path_vecs}
    aws s3 cp s3://wikibrain/w2v2/$lang/corpus.txt.bz2 ${path_vecs}/
    aws s3 cp s3://wikibrain/w2v2/$lang/dictionary.txt.bz2 ${path_vecs}/
    python36 -m wmf_embed.train.train_fasttext \
                --corpus ${path_vecs}/ \
                --output ${path_vecs}/$name \
                $extra_args 2>&1 | tee ${path_vecs}/log.txt
    pbzip2 ${path_vecs}/$name
    aws s3 cp ${path_vecs}/${name}.bz2 s3://wikibrain/w2v2/$lang/
    rm -rf ${path_vecs}/
}

export -f do_lang

echo $langs | tr ',' '\n' | parallel -j 6 --line-buffer do_lang $name $script_args '{}'

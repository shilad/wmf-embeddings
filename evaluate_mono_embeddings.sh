#!/bin/bash
#
# Rebuilds fasttext models models for a particular language
#
#  Usage: evaluate_mono_embeddings.sh           \
#         [--languages en,de,...]               \
#         [--name vectors.fasttext.txt]         \
#         [-- extra arguments to script]
#

set -e
set -x

languages="en,de,simple,ar,az,bg,ca,cs,da,eo,es,et,eu,fa,fi,fr,gl,he,hi,hr,hu,id,it,ja,kk,ko,lt,ms,nl,nn,no,pl,pt,ro,ru,sk,sl,sr,sv,tr,uk,vi,vo,war,zh"
algorithm=fasttext
name=vectors.fasttext.txt
script_args=""
jobs=1

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --languages)
        languages="$2"
        shift # past argument
        shift # past value
        ;;
        --name)
        name="$2"
        shift # past argument
        shift # past value
        ;;
        --jobs)
        jobs="$2"
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
        Unknown option '$key'
        Usage: [--languages en,de,...]
               [--name vectors.fasttext.txt]
               [--jobs number of parallel jobs]
               [-- extra arguments to script]
       " >&2
       exit 1
        ;;
    esac
done


echo evaluating model $name for languages $languages with extra args of $script_args >&2


function do_lang() {
    name=$1
    lang=$2
    shift
    shift
    extra_args=$@
    path_vecs=./vecs/${name}/${lang}
    mkdir -p ${path_vecs}
#    aws s3 cp s3://wikibrain/w2v2/$lang/${name}.bz2 ${path_vecs}/
#    pbunzip2 ${path_vecs}/${name}.bz2
    python3 -m wmf_embed.model.evaluate_monolingual_embeddings \
                --path ${path_vecs}/${name} \
                --lang ${lang} \
                $extra_args 2>&1 | tee ${path_vecs}/log.txt
#    pbzip2 ${path_vecs}/$name
#    aws s3 cp ${path_vecs}/${name}.bz2 s3://wikibrain/w2v2/$lang/
#    rm -rf ${path_vecs}/
}

export -f do_lang

for lang in $languages; do
    do_lang $name $lang $script_args
done

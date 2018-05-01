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

s3_base=s3://wikibrain/w2v3
languages="en,de,simple,ar,az,bg,ca,cs,da,eo,es,et,eu,fa,fi,fr,gl,he,hi,hr,hu,id,it,ja,kk,ko,lt,ms,nl,nn,no,pl,pt,ro,ru,sk,sl,sr,sv,tr,uk,vi,vo,war,zh"
algorithm=fasttext
name=vectors.fasttext.txt
script_args=""
jobs=6

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
        --s3_base)
        s3_base="$2"
        shift # past argument
        shift # past value
        ;;
        --jobs)
        jobs="$2"
        shift # past argument
        shift # past value
        ;;
        --algorithm)
        algorithm="$2"
        if [ $algorithm != "fasttext" ] && [ $algorithm != "doc2vec" ]; then
            echo "unknown algorithm: $algorithm. must be fasttext or doc2vec" >&2
            exit 1
        fi
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
               [--algorithm fasttext|doc2vec]
               [--jobs number of parallel jobs]
               [--s3_base s3 base directory (defaults to $s3_base) ]
               [-- extra arguments to script]
       " >&2
       exit 1
        ;;
    esac
done


echo building $algorithm model $name for languages $languages with extra args of $script_args >&2


function do_lang() {
    name=$1
    lang=$2
    algorithm=$3
    s3_base=$4
    shift
    shift
    shift
    shift
    extra_args=$@

    PYTHON3=
    for py3 in python3.6 python36 python3.5 python35 python3; do
        if which $py3; then
            PYTHON3=$py3
            break
        fi
    done
    if [ -z "$PYTHON3" ]; then
        echo "Couldn't find python3 interpreter!" >&2
        exit 1
    fi

    path_vecs=./vecs/${name}/${lang}
    mkdir -p ${path_vecs}
    aws s3 cp ${s3_base}/$lang/corpus.txt.bz2 ${path_vecs}/
    aws s3 cp ${s3_base}/$lang/dictionary.txt.bz2 ${path_vecs}/
    $PYTHON3 -m wmf_embed.train.train_${algorithm} \
                --corpus ${path_vecs}/ \
                --output ${path_vecs}/$name \
                $extra_args 2>&1 | tee ${path_vecs}/log.txt
    pbzip2 ${path_vecs}/$name
    aws s3 cp ${path_vecs}/${name}.bz2 ${s3_base}/$lang/
    rm -rf ${path_vecs}/
}

export -f do_lang

if [[ $languages = *","* ]]; then
    echo $languages |
    tr ',' '\n' |
    parallel -j ${jobs} --line-buffer do_lang $name '{}' $algorithm ${s3_base} $script_args
else
    do_lang $name $languages $algorithm ${s3_base} $script_args
fi


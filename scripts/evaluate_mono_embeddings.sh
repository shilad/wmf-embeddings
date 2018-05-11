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

if ! [ -d muse-eval/monolingual ]; then
    (cd muse-eval && ./get_evaluation.sh)
fi

languages="en,de,simple,es,fa,it"
algorithm=fasttext
name=vectors.fasttext.txt
s3_base=s3://wikibrain/w2v3
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
        --s3_base)
        s3_base="$2"
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


# find the right python
for py_exec in python3-6 python3.6 python3; do
    if which $py_exec; then
        break
    fi
done
echo "using python $py_exec"
export py_exec

function do_lang() {
    name=$1
    lang=$2
    shift
    shift
    extra_args=$@
    path_vecs=./vecs/${name}/${lang}
    mkdir -p ${path_vecs}
    aws s3 cp ${s3_base}/$lang/${name}.bz2 ${path_vecs}/
    pbunzip2 ${path_vecs}/${name}.bz2
    $py_exec -m wmf_embed.model.evaluate_monolingual_embeddings \
                --path ${path_vecs}/${name} \
                --lang ${lang} \
                $extra_args 2>&1 | tee ${path_vecs}/eval.mono.${name}.txt
    aws s3 cp ${path_vecs}/eval.mono.${name}.txt ${s3_base}/$lang/
    rm -rf ${path_vecs}/
}

export -f do_lang


if [[ $languages = *","* ]]; then
    echo $languages |
    tr ',' '\n' |
    parallel -j ${jobs} --line-buffer do_lang $name '{}' $script_args
else
    do_lang $name $languages $script_args
fi

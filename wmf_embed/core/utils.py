import numpy as np
import os.path
import scipy.spatial.distance

NP_FLOAT = 'float32'


def read_embeddings(titler, path, aligned=False):
    from .lang_embedding import LangEmbedding

    dims = None
    embeddings = []
    for wiki in os.listdir(path):
        subdir = os.path.join(path, wiki)
        if os.path.isdir(subdir) and (wiki == 'nav' or 'wiki' in wiki):
            lang = LangEmbedding(wiki, subdir, titler, aligned=aligned)
            if dims is None:
                dims = lang.dims()
            else:
                assert(lang.dims() == dims)
            embeddings.append(lang)

    return embeddings


def wiki_to_project(wiki):
    if wiki.endswith('wiki'):
        return wiki[:-4] + '.wikipedia'
    else:
        return wiki + '.wikipedia'

def project_to_wiki(project):
    if project.endswith('.wikipedia'):
        return project.split('.')[0] + 'wiki'
    else:
        return project


def max_cores():
    return min(max(1, os.cpu_count()), 8)


def nearest_neighbors(matrix, vector, n):
    v = vector.reshape(1, -1)
    dists = scipy.spatial.distance.cdist(matrix, v, 'cosine').reshape(-1)
    indexes = np.argsort(dists)
    return indexes.tolist()[:n]

def wiki_dirs(path, include_nav=False):
    result = []

    for fn in os.listdir(path):
        full_path = os.path.join(path, fn)
        if os.path.isdir(full_path):
            if include_nav and fn == 'nav':
                result.append(full_path)
            elif fn.endswith('wiki'):
                result.append(full_path)

    return result

def read_ids(path):
    with open(path, encoding='utf-8') as f:
        return [ line.strip() for line in f ]
from functools import reduce
from cleu.embeddings import Embeddings,Embedding
import numpy as np
import umap

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def create_similarity_matrix(list_embedding_src  ,list_embedding_tgt ,distance_function='cosine',csls_k=5):
    """Return 2d array of similarity matrix

    Args:
        list_embedding_src (list[Embedding]): Source list embedding
        list_embedding_tgt (list[Embedding]): Target list embedding
        distance_function (str, optional): Distance similarity. Defaults to 'cosine'.
        csls_k (int, optional): [description]. Defaults to 5.

    Returns:
        list[list[float]] : 2d array containing similarity 
    """
    similarity_matrix =[]
    for src_embedding in list_embedding_src:
        similarity_matrix.append(list() )
        for tgt_embedding in list_embedding_tgt:
            if distance_function=='cosine':
                similarity = src_embedding.compare_cosine(tgt_embedding)
            elif distance_function=='csls':
                similarity = src_embedding.compare_csls(tgt_embedding,csls_k=csls_k)
            similarity_matrix[-1].append(similarity)
    return similarity_matrix

def reduce_dimensionality_2d(embedding_matrix, dimensionality_reduction='umap'):
    if (dimensionality_reduction=='umap'):
        reducer = umap.UMAP()
    elif (dimensionality_reduction=='pca'):
        reducer = PCA(n_components=2)

    elif (dimensionality_reduction=='tsne'):
        reducer = TSNE(n_components=2)
    else:
        ValueError('Invalid Dimensionality Reduction method ')
    embeddings_2d = reducer.fit_transform(embedding_matrix)
    return embeddings_2d



def combine_embeddings_matrix(list_embeddings):
    """This function accept list of Embeddings object and return the combined embeddings matrix

    Args:
        list_embeddings (list[Embeddings]): List of Embeddings to combine
    """
    combine_matrix = map(lambda embeddings : embeddings.embeddings_matrix,list_embeddings)
    embeddings_matrix = np.vstack(combine_matrix)
    return embeddings_matrix


def combine_embeddings_language(list_embeddings):
    """This function accept list of Embeddings object and return the combined embeddings language

    Args:
        list_embeddings (list[Embeddings]): List of Embeddings to combine
    """
    languages = []
    for embeddings in list_embeddings:
        languages =  languages + ([embeddings.lang] *len(embeddings.id2word) )
    return languages


def combine_embeddings_word(list_embeddings):

    """This function accept list of Embeddings object and return the combined embeddings words

    Args:
        list_embeddings (list[Embeddings]): List of Embeddings to combine
    """
    list_word =  []
    for embeddings in list_embeddings:
        list_word = list_word+ embeddings.id2word
    return list_word
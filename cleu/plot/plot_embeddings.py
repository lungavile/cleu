import seaborn as sns
from cleu.embeddings import Embedding,Embeddings
from cleu.utils import utils
import matplotlib.pyplot as plt
import altair as alt
import pandas as pd
import numpy as np

def plot_similarity(list_embedding_src,list_embedding_tgt,distance_function='cosine',csls_k=5):
    assert(len(list_embedding_src) > 0 and len(list_embedding_tgt) >0)
    assert (len(list_embedding_src) == len(list_embedding_tgt))


    words_1 = list(map(lambda embedding: embedding.word , list_embedding_src))
    words_2 = list(map(lambda embedding: embedding.word , list_embedding_tgt))
    
    sns.set(rc={'figure.figsize':(13.7,10.27), 'image.cmap' :  "Blues" })

    similarity_matrix = utils.create_similarity_matrix(list_embedding_src,list_embedding_tgt,distance_function,csls_k)
    plot_title = 'Cosine' if distance_function =='cosine' else 'Cross-domain Similarity Local Scaling'
    ax = sns.heatmap(similarity_matrix ,annot=True,xticklabels=words_2, yticklabels=words_1)
    ax.set_title(plot_title)
    ax.xaxis.set_ticks_position('top') 
    plt.show()

def plot_embeddings_2d(list_embeddings,width,height,dimensionality_reduction='umap'):
    """Plot embeddings that are reduced to 2d

    Args:
        list_embeddings (list[Embeddings]): List of Embeddings to plot
        width (int): Plot width
        height (int): Plot height
        dimensionality_reduction (str, optional): ('umap','pca,'tsne'). Defaults to 'umap'.

    """
    assert(len(list_embeddings) >0)
    embeddings_matrix = utils.combine_embeddings_matrix(list_embeddings)
    list_word =  utils.combine_embeddings_word(list_embeddings)
    color = utils.combine_embeddings_language(list_embeddings)
    # assert(len(list_word) <=5000)    
    embeddings_2d = utils.reduce_dimensionality_2d(embeddings_matrix,dimensionality_reduction)
    plot_df = pd.DataFrame(
            {
                "X-Axis": embeddings_2d[:,0],
                "Y-Axis": embeddings_2d[:,1],
                "name": list_word,
                "Language" : color 
            }
        )
    plot = alt.Chart(plot_df).mark_circle(size=60).encode(
        x='X-Axis',
        y='Y-Axis',
        tooltip=["name"],
        color= 'Language',
    ).properties(
        width=width,
        height=height
    )

    text = (
        alt.Chart(plot_df)
        .mark_text(dx=-15, dy=3, color="black")
        .encode(
            x="X-Axis",
            y="Y-Axis",
            text="name",
        )
    )
    plot =plot.interactive()
    plot = plot + text
    return plot

def plot_embeddings_neighbours(
    list_embedding,
    list_tgt_embeddings,
    width,
    height,
    dimensionality_reduction='umap',
    k=5,
    distance_function='cosine',
    csls_k=10
):
    """Accept list of Embedding as query and plot its target nearest neighbours

    Args:
        list_embedding (list[Embedding]): List of Embedding as Query
        list_tgt_embeddings (list[Embeddings]): List of Embeddings as Target
        width (int): Plot width
        height (int): Plot height
        dimensionality_reduction (str, optional): ('umap','pca,'tsne'). Defaults to 'umap'.    
        k (int, optional): Number of neighbours returned. Defaults to 5.
        distance_function (str, optional): Distance function that will be used to measure two different embedding vectors. csls or cosine(Default). 
        csls_k (int, optional): Number of neighbours that will be used for CSLS mean similarity. Defaults to 10.
    """
    # assert((len(list_tgt_embeddings) >0), "Target Embeddings must not be empty")
    topic = []
    lang = []
    neighbours_embeddings = []
    for tgt_embeddings in list_tgt_embeddings:
        for embedding in list_embedding:
            similarities,neighbours  = tgt_embeddings.get_nearest_neighbours(embedding,k,distance_function,csls_k)
            neighbours_embeddings =  neighbours_embeddings + neighbours
            topic = topic + ([embedding.word] * len(neighbours) )
            lang = lang + ([tgt_embeddings.lang] * len(neighbours))
    embeddings_matrix = list(map(lambda embedding: embedding.vector,neighbours_embeddings))
    list_word =  list(map(lambda embedding: embedding.word,neighbours_embeddings))
    color = topic
    assert(len(list_word) <=5000)    
    embeddings_2d = utils.reduce_dimensionality_2d(embeddings_matrix,dimensionality_reduction)
    plot_df = pd.DataFrame(
            {
                "X-Axis": embeddings_2d[:,0],
                "Y-Axis": embeddings_2d[:,1],
                "name": list_word,
                "language" :  lang,
                "Topic" : color 
            }
        )
    plot = alt.Chart(plot_df).mark_circle(size=60).encode(
        x='X-Axis',
        y='Y-Axis',
        tooltip=["name","language"],
        color= 'Topic',
    ).properties(
        width=width,
        height=height
    )

    text = (
        alt.Chart(plot_df)
        .mark_text(dx=-15, dy=3, color="black")
        .encode(
            x="X-Axis",
            y="Y-Axis",
            text="name",
        )
    )
    plot =plot.interactive()
    plot = plot + text
    return plot



from typing import Tuple
from faiss.swigfaiss import simd_histogram_16
import numpy as np
import faiss

class Embedding:
    def __init__(self,lang,dim,word,id,vector):
        self.lang =lang
        self.dim= dim
        self.word =word
        self.id = id
        self.vector = vector



# tetap pakai class embeddings
#  bedanya waktu get nearest neighbours lempar object embedding
class Embeddings:
    def __init__(self,dim):
        self.lang =""
        self.dim= dim
        self.word2id = {}
        self.id2word= []
        self.vector = []
        self.build_faiss_index()
    
    def build_faiss_index(self,cuda=False):
        self.embedding_index = faiss.IndexFlatL2(self.dim)

    def load_embeddings(self,path,lang,ext='txt',max_vocab=10000 ):
        available_ext = ['txt','pck']
        assert(ext in available_ext)
        if ext=='txt':
            self.load_txt(path,lang,max_vocab)
        elif ext=='pck':
            print('pck')
        # TODO implements load txt and ppth
    
    def load_txt(self,path,lang,max_vocab):
        with open(path,'r') as f:
            for i,line in enumerate(f):
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                self.id2word.append(word)
                self.word2id[word] = len(self.word2id)
                self.vector.append(embedding)
                if i+1 >= max_vocab:
                    break
            self.embedding_index.add( np.asarray(self.vector).astype(np.float32) )
    
    def getEmbeddingById(self,id:int)->Embedding:
        embedding = Embedding(id=id,word = self.id2word[id], dim=self.dim,vector=self.vector[id] ,lang=self.lang)
        return embedding

    def getEmbeddingByWord(self,word:str)->Embedding:
        id = self.word2id[word]
        embedding = Embedding(id=id,word = word, dim=self.dim,vector=self.vector[id] ,lang=self.lang)
        return embedding

        

    def getNearestNeighbours(self,embedding : Embedding,k=5,distance_function='cosine') -> Tuple[list[np.int64],list[Embedding]]:
        if distance_function=='cosine':
            similarities, indices = self.get_neighbours_cosine(embedding,k=5)
        # elif distance_function=='csls':
        
        # faiss returns an 2d array instead 1
        embedding_list  = list(map(lambda index :  self.getEmbeddingById(index), indices))
        return similarities,embedding_list

    def get_neighbours_cosine(self,embedding : Embedding,k=5 )-> Tuple[list[np.float32],list[np.int64]]:
        similarities, indices = self.embedding_index.search(np.array([embedding.vector]).astype(np.float32), k) # sanity check
        return similarities[0],indices[0]


    # def get_neighbours_csls(self,vector,k=5):
    

    # def transform_embeddings(self,mapping_matrix : MappingMatrix):
    #     self.vector = self.vector @ mapping_matrix.transformation_matrix
    
    # def get_cross

    # def get_nearest_neighbour(self,keyword,k=5):
    #     pass

    # def get_cross_domain_neighbour(self,keyword,k=5):
    #     pass

    
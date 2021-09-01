from typing import Tuple
import numpy as np
import faiss

class Embedding:
    def __init__(self,lang,dim,word,id,vector,embeddings):
        self.lang =lang
        self.dim= dim
        self.word =word
        self.id = id
        self.vector = vector
        # Used to reference parents
        self.embeddings =embeddings
    
    def __str__(self):
        return self.word + " , " + self.lang

    def compare_cosine(self,target_embedding):
        return np.dot( self.vector,target_embedding.vector)
    
    
    def compare_csls(self,target_embedding,csls_k):
        src= self.embeddings
        tgt= target_embedding.embeddings

        src.build_mean_similarity(src,tgt,csls_k)
        cosine_source_target= self.compare_cosine(target_embedding)
        csls = (cosine_source_target*2) - src.get_mean_similarity_by_word(self.word,tgt.lang,csls_k=csls_k) - tgt.get_mean_similarity_by_word(target_embedding.word,self.lang,csls_k=csls_k)
            
        return csls



# tetap pakai class embeddings
#  bedanya waktu get nearest neighbours lempar object embedding
class Embeddings:
    def __init__(self,lang,dim):
        self.lang =lang
        self.dim= dim
        self.word2id = {}
        self.id2word= []
        self.vector = []
        
        self.mean_similarity = {}
    
    def build_faiss_index(self,cuda=False):
        if cuda==False:
            self.embedding_index = faiss.IndexFlatIP(self.dim)
        else:
            res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = 0
            index = faiss.GpuIndexFlatIP(res, self.dim, config)
        self.embedding_index.add( np.asarray(self.vector).astype(np.float32) )


    def load_embeddings(self,path,lang,ext='txt',max_vocab=10000):
        available_ext = ['txt','pck']
        assert(ext in available_ext)
        if ext=='txt':
            self.load_txt(path,lang,max_vocab)
        elif ext=='pck':
            print('pck')
        # TODO implements load txt and ppth
    
    def load_txt(self,path,lang,max_vocab):
        if len(self.vector) !=0:
            print("Embeddings already exists, create new object instead changing old ones")
            return
        self.lang=lang
        with open(path,'r') as f:
            for i,line in enumerate(f):
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float32)
                if len(embedding) != 300:
                    print(f"Different dimension occured in line ${i}")
                    continue
                self.id2word.append(word)
                self.word2id[word] = len(self.word2id)
                self.vector.append(embedding)
                if len(self.id2word) >= max_vocab:
                    break
        self.vector= np.array(self.vector)
        self.vector = self.vector / np.linalg.norm(self.vector,ord=2,axis=1,keepdims=True)
        self.build_faiss_index()
    
    def getEmbeddingById(self,id:int)->Embedding:
        embedding = Embedding(id=id,word = self.id2word[id], dim=self.dim,vector=self.vector[id] ,lang=self.lang,embeddings=self)
        return embedding

    def getEmbeddingByWord(self,word:str)->Embedding:
        id = self.word2id[word]
        embedding = Embedding(id=id,word = word, dim=self.dim,vector=self.vector[id] ,lang=self.lang,embeddings=self)
        return embedding

        
    def getNearestNeighbours(self,embedding : Embedding,k=5,distance_function='cosine',csls_k=10) -> Tuple[list[np.int64],list[Embedding]]:
        if distance_function=='cosine':
            similarities, indices = self.get_neighbours_cosine(embedding,k=5)
        elif distance_function=='csls':
            if embedding.lang== self.lang:
                print("CSLS is intended to works on different languages")
                raise ValueError
            similarities, indices = self.get_neighbours_csls(embedding,k=5,csls_k=csls_k)
            
        # faiss returns an 2d array instead 1
        embedding_list  = list(map(lambda index :  self.getEmbeddingById(index), indices))
        return similarities,embedding_list

    def get_neighbours_cosine(self,embedding : Embedding,k=5 )-> Tuple[list[np.float32],list[np.int64]]:
        similarities, indices = self.embedding_index.search(np.array([embedding.vector]).astype(np.float32), k) # sanity check
        return similarities[0],indices[0]

    def build_mean_similarity(self,src,tgt,csls_k=10):
        src_dict = src.lang + f"top_k_{csls_k}"
        tgt_dict = tgt.lang + f"top_k_{csls_k}"
        # tgt->src
        if src_dict not in tgt.mean_similarity:
            tgt_src_similarities, tgt_src_indices = src.embedding_index.search(np.array(tgt.vector).astype(np.float32) , csls_k)
            tgt_src_similarities= tgt_src_similarities.mean(1)
            tgt.mean_similarity[src_dict ] =  tgt_src_similarities
        # src->tgt
        if tgt_dict not in src.mean_similarity:
            src_tgt_similarities, src_tgt_indices =  tgt.embedding_index.search(np.array(src.vector).astype(np.float32) , csls_k)
            src_tgt_similarities= src_tgt_similarities.mean(1)
            src.mean_similarity[tgt_dict] = src_tgt_similarities
        

    def get_neighbours_csls(self,embedding:Embedding,k=5,csls_k=10):
        src= self
        tgt= embedding.embeddings

        self.build_mean_similarity(src,tgt,csls_k)

        scores=[]
        for src_word in src.id2word:
            vec_src= src.getEmbeddingByWord(src_word).vector
            vec_tgt = embedding.vector
            cosine_source_target = np.dot(vec_src,vec_tgt)
            csls = (cosine_source_target*2) - src.get_mean_similarity_by_word(src_word,embedding.lang,csls_k) - tgt.get_mean_similarity_by_word(embedding.word,self.lang,csls_k)
            scores.append(csls)
        scores = np.array(scores)
        top_k_csls = scores.argsort()[-k:][::-1]
        return scores[top_k_csls], top_k_csls


        # Return CSLS formula
    def get_mean_similarity_by_word(self,word,lang,csls_k):        
        dict = lang + f"top_k_{csls_k}"
        id = self.word2id[word]
        return self.mean_similarity[dict][id]
    
    def get_mean_similarity_by_id(self,id,lang,csls_k):
        dict = lang + f"top_k_{csls_k}"
        return self.mean_similarity[dict][id]
    
    # def transform_embeddings(self,mapping_matrix : MappingMatrix):
    #     self.vector = self.vector @ mapping_matrix.transformation_matrix
    
    # def get_cross

    # def get_nearest_neighbour(self,keyword,k=5):
    #     pass

    # def get_cross_domain_neighbour(self,keyword,k=5):
    #     pass

    
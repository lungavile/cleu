from typing import Tuple
import numpy as np
import faiss

# https://samnicholls.net/2016/06/15/how-to-sphinx-readthedocs/

class Embedding:
    def __init__(self,lang,dim,word,id,vector,embeddings):
        self.lang =lang
        self.dim= dim
        self.word =word
        self.id = id
        self.vector = vector
        # Used to reference parents
        self.embeddings =embeddings
    

    def compare_cosine(self,target_embedding):
        """Compare and return cosine similarity from two Embedding object

        :param target_embedding (Embedding): Embedding object that will be compared

        Returns:
            np.int64: Similarity that are measured using cosine. Returned values will range from 0 to 1
        
        """
        return np.dot( self.vector,target_embedding.vector)
    
    
    def compare_csls(self,target_embedding,csls_k):
        """Compare and return cosine similarity from two Embedding object

        :param target_embedding (Embedding): Embedding object that will be compared.
        :param csls_k (int, optional): Number of neighbours that will be used for CSLS mean similarity. Defaults to 10.
        Returns:
            np.int64: Similarity that are measured using cosine. Returned values will range from -2 to 2
        
        """
        src= self.embeddings
        tgt= target_embedding.embeddings

        src.build_mean_similarity(src,tgt,csls_k)
        cosine_source_target= self.compare_cosine(target_embedding)
        csls = (cosine_source_target*2) - src.get_mean_similarity_by_word(self.word,tgt.lang,csls_k=csls_k) - tgt.get_mean_similarity_by_word(target_embedding.word,self.lang,csls_k=csls_k)
            
        return csls

class Embeddings:
    
    def __init__(self,lang,dim,cuda):
        self.lang =lang
        self.dim= dim
        self.cuda =cuda
        self.word2id = {}
        self.id2word= []
        self.embeddings_matrix = []
        self.mean_similarity = {}
    

    def build_faiss_index(self,cuda=False):
        """This function build faiss index, for searching fast nearest neighbour by using ANN (Approximation Nearest Neighbours)
        
        Args:
            cuda (bool, optional): Use GPU or not for indexing Defaults to False.
        
        """
        if cuda==False:
            self.embedding_index = faiss.IndexFlatIP(self.dim)
        else:
            res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = 0
            index = faiss.GpuIndexFlatIP(res, self.dim, config)
        self.embedding_index.add( np.asarray(self.embeddings_matrix).astype(np.float32) )


    def load_embeddings(self,path:str,lang:str,ext='txt',max_vocab=10000):
        """This function are used to load word embeddings,currently only txt formats are supported

        Args:
            path (str): Path to the word embeddings file.
            lang (str): Word embeddings language.
            ext (str, optional): File formats, currently only support txt.
            max_vocab (int, optional): Maximum word embedding to load defaults to 10000.
        
        """
        available_ext = ['txt','pck']
        assert(ext in available_ext)
        if ext=='txt':
            self.load_txt(path,lang,max_vocab)
        elif ext=='pck':
            print('pck')
        # TODO implements load txt and ppth
    
    def load_txt(self,path,lang,max_vocab):
        """This function are used to load word embeddings that are using text-file format

        Args:
            path (str): Path to the word embeddings file
            lang (str): Word embeddings language
            max_vocab (int, optional): Maximum word embedding to load defaults to 10000.
        
        """
        if len(self.embeddings_matrix) !=0:
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
                self.embeddings_matrix.append(embedding)
                if len(self.id2word) >= max_vocab:
                    break
        self.embeddings_matrix= np.array(self.embeddings_matrix)
        self.embeddings_matrix = self.embeddings_matrix / np.linalg.norm(self.embeddings_matrix,ord=2,axis=1,keepdims=True)
        self.build_faiss_index(self.cuda)
    
    def get_embedding_by_id(self,id:int)->Embedding:
        """Get embedding object by id

        Args:
            id (int): Embedding id

        Returns:
            Embedding: Associated Embedding Object
        
        """
        embedding = Embedding(id=id,word = self.id2word[id], dim=self.dim,vector=self.embeddings_matrix[id] ,lang=self.lang,embeddings=self)
        return embedding

    def get_embedding_by_word(self,word:str)->Embedding:
        """Get embedding object by word

        Args:
            word (str): Embedding word

        Returns:
            Embedding: Associated Embedding Object
        
        """
        id = self.word2id[word]
        embedding = Embedding(id=id,word = word, dim=self.dim,vector=self.embeddings_matrix[id] ,lang=self.lang,embeddings=self)
        return embedding

        
    def get_nearest_neighbours(self,embedding : Embedding,k=5,distance_function='cosine',csls_k=10):
        """This function are used to get nearest neighbour from the monolingual or cross-lingual Embeddings. There are two distance function currently supported, cosine and cross-domain local similarity scaling
        
        Args:
            embedding (Embedding): Embedding that will be used as query
            k (int, optional): Number of neighbours returned. Defaults to 5.
            distance_function (str, optional): Distance function that will be used to measure two different embedding vectors. csls or cosine(Default). 
            csls_k (int, optional): Number of neighbours that will be used for CSLS mean similarity. Defaults to 10.
        
        Raises:
            ValueError: CSLS different values
        
        Returns:
            Tuple[list[np.int64],list[Embedding]]: [Return list of the similarities and the neighbours embedding]
        
        """
        if distance_function=='cosine':
            similarities, indices = self.get_neighbours_cosine(embedding,k=k)
        elif distance_function=='csls':
            if embedding.lang== self.lang:
                print("CSLS is intended to works on different languages")
                raise ValueError
            similarities, indices = self.get_neighbours_csls(embedding,k=k,csls_k=csls_k)
            
        # faiss returns an 2d array instead 1
        embedding_list  = list(map(lambda index :  self.get_embedding_by_id(index), indices))
        return similarities,embedding_list

    def get_neighbours_cosine(self,embedding : Embedding,k=5 )-> Tuple[list[np.float32],list[np.int64]]:
        """This function are used to get nearest neighbour from the monolingual or cross-lingual Embeddings by using cosine similarity

        Args:
            embedding (Embedding): Embedding that will be used as query
            k (int, optional): Number of neighbours returned. Defaults to 5.
            

        Returns:
            Tuple[list[np.int64],list[int]]: [Return list of the similarities and the neighbours index]
        
        """
        similarities, indices = self.embedding_index.search(np.array([embedding.vector]).astype(np.float32), k) # sanity check
        return similarities[0],indices[0]

    def build_mean_similarity(self,src,tgt,csls_k=10):
        """This function are used to build the CSLS mean similarity in source and target space 

        Args:
            src (Embeddings): Source word embeddings
            tgt (Embeddings): Target word embeddings
            csls_k (int, optional): Number of neighbours that will be used for CSLS mean similarity. Defaults to 10.
        
        """
        src_dict = src.lang + f"top_k_{csls_k}"
        tgt_dict = tgt.lang + f"top_k_{csls_k}"
        # tgt->src
        if src_dict not in tgt.mean_similarity:
            tgt_src_similarities, tgt_src_indices = src.embedding_index.search(np.array(tgt.embeddings_matrix).astype(np.float32) , csls_k)
            tgt_src_similarities= tgt_src_similarities.mean(1)
            tgt.mean_similarity[src_dict ] =  tgt_src_similarities
        # src->tgt
        if tgt_dict not in src.mean_similarity:
            src_tgt_similarities, src_tgt_indices =  tgt.embedding_index.search(np.array(src.embeddings_matrix).astype(np.float32) , csls_k)
            src_tgt_similarities= src_tgt_similarities.mean(1)
            src.mean_similarity[tgt_dict] = src_tgt_similarities
        

    def get_neighbours_csls(self,embedding:Embedding,k=5,csls_k=10):
        """This function are used to get nearest neighbour from the monolingual or cross-lingual Embeddings by using cross-domain local similarity scaling

        Args:
            embedding (Embedding): Embedding that will be used as query
            k (int, optional): Number of neighbours returned. Defaults to 5.
            

        Returns:
            Tuple[list[np.int64],list[int]]: [Return list of the similarities and the neighbours index]
        
        """
        src= self
        tgt= embedding.embeddings

        self.build_mean_similarity(src,tgt,csls_k)

        scores=[]
        for src_word in src.id2word:
            vec_src= src.get_embedding_by_word(src_word).vector
            vec_tgt = embedding.vector
            cosine_source_target = np.dot(vec_src,vec_tgt)
            csls = (cosine_source_target*2) - src.get_mean_similarity_by_word(src_word,embedding.lang,csls_k) - tgt.get_mean_similarity_by_word(embedding.word,self.lang,csls_k)
            scores.append(csls)
        scores = np.array(scores)
        top_k_csls = scores.argsort()[-k:][::-1]
        return scores[top_k_csls], top_k_csls


    def get_mean_similarity_by_word(self,word,lang,csls_k):
        """This function are used to get the mean similarity value by embedding word

        Args:
            word (str): Embedding word
            lang (str): Target language
            csls_k (int, optional): Number of neighbours that will be used for CSLS mean similarity. Defaults to 10.

        Returns:
            np.int64: Return the mean similarity associated embedding
        
        """
        dict = lang + f"top_k_{csls_k}"
        id = self.word2id[word]
        return self.mean_similarity[dict][id]
    
    def get_mean_similarity_by_id(self,id,lang,csls_k):
        """This function are used to get the mean similarity value by embedding id

        Args:
            id (int): Embedding Id
            lang (str): Target Language
            csls_k (int, optional): Number of neighbours that will be used for CSLS mean similarity. Defaults to 10.

        Returns:
            np.int64: Return the mean similarity associated embedding
        
        """
        dict = lang + f"top_k_{csls_k}"
        return self.mean_similarity[dict][id]


    
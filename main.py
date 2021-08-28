from cleu.embeddings import Embeddings

clwe_en = Embeddings(dim=300)
clwe_en.load_embeddings('data/vectors-en.txt',lang='en',max_vocab=1000)



clwe_id = Embeddings(dim=300)
clwe_id.load_embeddings('data/vectors-id.txt',lang='id',max_vocab=1000)

emb_you = clwe_en.getEmbeddingByWord('you')
emb_saya = clwe_id.getEmbeddingByWord('saya')
similarity,neighbours_saya = clwe_en.getNearestNeighbours(emb_saya,k=5)
for emb in neighbours_saya:
    print(emb.word)

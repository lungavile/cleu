from cleu import plot
from cleu.embeddings import Embeddings
from cleu.plot import plot_embeddings

clwe_en = Embeddings(dim=300,lang='en')
clwe_en.load_embeddings('data/vectors-en.txt',lang='en',max_vocab=1000)



clwe_id = Embeddings(dim=300,lang='id')
clwe_id.load_embeddings('data/vectors-id.txt',lang='id',max_vocab=1000)

# emb_you = clwe_en.getEmbeddingByWord('you')
# emb_saya = clwe_id.getEmbeddingByWord('kamu')
# print(emb_you.compare_csls(emb_saya,csls_k=5))

# similarity,neighbours_you = clwe_id.getNearestNeighbours(emb_you,distance_function='csls',k=5,csls_k=3)
# for emb in neighbours_you:
#     print(emb.word)
# for sim in similarity:
#     print(sim)

# words_id = ["makan","memakan","minum","jalan","hipotesis","kucing","belajar","uang","polisi","obligasi"]
# words_en = ["eat","eating","drink","walk","hypothesis","cat","study","money","police","bonds"]
# emb_id = list(map(lambda word : clwe_id.getEmbeddingByWord(word) ,words_id))
# emb_en = list(map(lambda word : clwe_en.getEmbeddingByWord(word) ,words_en))

# plot_embeddings.plot_confusion_similarity(emb_id,emb_en,distance_function='csls',csls_k=3)

plot_embeddings.plot_embeddings_2d(clwe_id)

# ubah return jadi object di nearest neighbour [DONE]
# visualisasi topik
# tambah cosine dan csls antara 2 embedding obj [DONE]
# visualisasi graph
# visualisasi confusion [Done]
# tambah validasi
# tambah visualisasi
# tambah dokumentasi
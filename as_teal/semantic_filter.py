from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import gensim
import numpy as np
import warnings


def load_wordset(path):
    file = open(path)
    wordset = set()
    while 1:
        line = file.readline()
        if not line:
            break
        line = line.replace('\n', '')
        items = line.split("\t")
        hypo = items[0]
        wordset.add(hypo)
    file.close()
    return wordset


def filter_dataset(path, new_path, k_means, word_vectors, tau):
    centroids=k_means.cluster_centers_
    file = open(path)
    out_file=open(new_path,'w+')
    while 1:
        line = file.readline()
        if not line:
            break
        line = line.replace('\n', '')
        items = line.split("\t")
        hypo = items[0]

        find=False
        for i in range(0,len(centroids)):
            centroid=centroids[i]
            if cosine_similarity(centroid, word_vectors[hypo])>tau:
                find=True
                print(line)
                break
        if find:
            out_file.write(line+'\n')
    file.close()


#load glove
gensim_file = '../data/glove_model.txt'
word_vectors = gensim.models.KeyedVectors.load_word2vec_format(gensim_file, binary=False)  # GloVe Model
print('load model success')
emb_dim=100
warnings.filterwarnings('ignore')

train_word_set=load_wordset('train.txt')
test_word_set=load_wordset('test.txt')
total_word_set=train_word_set.union(test_word_set)

data_matrix=np.zeros(shape=(len(total_word_set),emb_dim))
i=0
for word in total_word_set:
    data_matrix[i]=word_vectors[word]
    i=i+1

k_means=KMeans(n_clusters=100)
k_means.fit(data_matrix)
print('cluster model successfully')

tau=0.6
filter_dataset('probase_pos.txt', 'as_probase_pos.txt', k_means, word_vectors, tau)
filter_dataset('probase_neg.txt', 'as_probase_neg.txt', k_means, word_vectors, tau)
import numpy as np
import gensim
import math
from keras.models import load_model

# load data
pos_list = list()
neg_list = list()
file = open('bless_binary.txt')
while 1:
    line = file.readline()
    if not line:
        break
    line = line.replace('\n', '')
    str = line.split('\t')
    hypo = str[0]
    hyper = str[1]
    if str[2] == '1':
        pos_list.append((hypo, hyper))
    else:
        neg_list.append((hypo, hyper))
file.close()
print(len(pos_list))
print(len(neg_list))

# load glove
gensim_file = 'word_vectors_u_teal.txt'
word_vectors = gensim.models.KeyedVectors.load_word2vec_format(gensim_file, binary=False)  # GloVe Model
print('load model successfully')

model = load_model('model_bless.h5')

# train basicnn
emb_dim = 100
# load input
positive_count = 0
for i in range(len(pos_list)):
    (hypo, hyper) = pos_list[i]
    if hypo not in word_vectors:
        continue
    if hyper not in word_vectors:
        continue
    positive_count = positive_count + 1
pos_data = np.zeros((positive_count, emb_dim))
pos_labels = np.zeros((positive_count, emb_dim))
positive_count = 0
for i in range(len(pos_list)):
    (hypo, hyper) = pos_list[i]
    if hypo not in word_vectors:
        continue
    if hyper not in word_vectors:
        continue
    pos_data[positive_count] = word_vectors[hypo]
    pos_labels[positive_count] = word_vectors[hyper]
    positive_count = positive_count + 1

negative_count = 0
for i in range(len(neg_list)):
    (hypo, hyper) = neg_list[i]
    if hypo not in word_vectors:
        continue
    if hyper not in word_vectors:
        continue
    negative_count = negative_count + 1
neg_data = np.zeros((negative_count, emb_dim))
neg_labels = np.zeros((negative_count, emb_dim))
negative_count = 0
for i in range(len(neg_list)):
    (hypo, hyper) = neg_list[i]
    if hypo not in word_vectors:
        continue
    if hyper not in word_vectors:
        continue
    neg_data[negative_count] = word_vectors[hypo]
    neg_labels[negative_count] = word_vectors[hyper]
    negative_count = negative_count + 1

# do model prediction
threshold = 0
accurate = 0
[pre_pos, pre_neg] = model.predict_on_batch(x=[pos_data])
for i in range(positive_count):
    pos_dist = np.linalg.norm(pos_labels[i] - pre_pos[i])
    neg_dist = np.linalg.norm(pos_labels[i] - pre_neg[i])
    prob = math.tanh(neg_dist - pos_dist)
    if prob >= threshold:
        accurate = accurate + 1

[pre_pos, pre_neg] = model.predict_on_batch(x=[neg_data])
for i in range(negative_count):
    pos_dist = np.linalg.norm(neg_labels[i] - pre_pos[i])
    neg_dist = np.linalg.norm(neg_labels[i] - pre_neg[i])
    prob = math.tanh(neg_dist - pos_dist)
    if prob < threshold:
        accurate = accurate + 1

accuracy = accurate / (positive_count + negative_count)
print(accuracy)

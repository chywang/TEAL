import numpy as np
import gensim
from keras.layers import Input, Dense
from keras.models import Model

# load data
pos_list = list()
neg_list = list()
file = open('probase_pos.txt')
while 1:
    line = file.readline()
    if not line:
        break
    line = line.replace('\n', '')
    str = line.split('\t')
    hypo = str[0]
    hyper = str[1]
    pos_list.append((hypo, hyper))
file.close()

file = open('probase_neg.txt')
while 1:
    line = file.readline()
    if not line:
        break
    line = line.replace('\n', '')
    str = line.split('\t')
    hypo = str[0]
    hyper = str[1]
    neg_list.append((hypo, hyper))
file.close()
print(len(pos_list))
print(len(neg_list))

# load glove
gensim_file = 'word_vectors_u_teal.txt'
word_vectors = gensim.models.KeyedVectors.load_word2vec_format(gensim_file, binary=False)  # GloVe Model
print('load model successfully')

# train basicnn
emb_dim = 100
# load input
positive_count = 0
for i in range(len(pos_list)):
    (hypo, hyper) = pos_list[i]
    positive_count = positive_count + 1
pos_data = np.zeros((positive_count, emb_dim))
pos_labels = np.zeros((positive_count, emb_dim))
positive_count = 0
for i in range(len(pos_list)):
    (hypo, hyper) = pos_list[i]
    pos_data[positive_count] = word_vectors[hypo]
    pos_labels[positive_count] = word_vectors[hyper]
    positive_count = positive_count + 1

negative_count = 0
for i in range(len(neg_list)):
    (hypo, hyper) = neg_list[i]
    negative_count = negative_count + 1
neg_data = np.zeros((negative_count, emb_dim))
neg_labels = np.zeros((negative_count, emb_dim))
negative_count = 0
for i in range(len(neg_list)):
    (hypo, hyper) = neg_list[i]
    neg_data[negative_count] = word_vectors[hypo]
    neg_labels[negative_count] = word_vectors[hyper]
    negative_count = negative_count + 1

# define model structure
input = Input(shape=(emb_dim,), name='input')
hidden = Dense(emb_dim*2, activation='tanh', name='hidden')(input)
pos_output = Dense(emb_dim, activation='linear', name='pos_output')(hidden)
neg_output = Dense(emb_dim, activation='linear', name='neg_output')(hidden)
model = Model(inputs=[input], outputs=[pos_output, neg_output])
for i in range(5):
    model.compile(optimizer='adam', loss='mse',
                  loss_weights=[1, 0])
    model.fit([pos_data], [pos_labels, pos_labels],
              epochs=50, batch_size=128)
    model.compile(optimizer='adam', loss='mse',
                  loss_weights=[0, 1])
    model.fit([neg_data], [neg_labels, neg_labels],
              epochs=50, batch_size=128)
model.save('model.h5')

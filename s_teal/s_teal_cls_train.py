import numpy as np
import gensim
from keras.models import load_model
from sklearn.svm import SVC
from sklearn.externals import joblib

# load data
pos_list = list()
neg_list = list()
file = open('train.txt')
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

# load glove
gensim_file = 'word_vectors_s_teal.txt'
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

model = load_model('proj_model.h5')
[pre_pos, pre_neg] = model.predict_on_batch(x=[pos_data])

[pre_pos, pre_neg] = model.predict_on_batch(x=[pos_data])
pos_svm_data = np.column_stack(
    (pos_data, pos_labels, pos_data - pos_labels, pre_pos - pos_labels, pre_neg - pos_labels))
pos_svm_label = np.ones((positive_count, 1))

[pre_pos, pre_neg] = model.predict_on_batch(x=[neg_data])
neg_svm_data = np.column_stack(
    (neg_data, neg_labels, neg_data - neg_labels, pre_pos - neg_labels, pre_neg - neg_labels))
neg_svm_label = np.zeros((negative_count, 1))

model = SVC(kernel='rbf', degree=2)
X = np.row_stack((pos_svm_data, neg_svm_data))
y = np.row_stack((pos_svm_label, neg_svm_label)).reshape(positive_count + negative_count, )
model.fit(X, y)
joblib.dump(model, 'cls.pickle')

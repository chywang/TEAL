import numpy as np
import gensim
from keras.layers import Input, Dense, Concatenate
from keras.models import Model


def load_training_set(path):
    # load training data
    pos_list = list()
    neg_list = list()
    file = open(path)
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
    return pos_list, neg_list


def load_probase_training_set(path):
    data_list = list()
    file = open(path)
    while 1:
        line = file.readline()
        if not line:
            break
        line = line.replace('\n', '')
        str = line.split('\t')
        hypo = str[0]
        hyper = str[1]
        data_list.append((hypo, hyper))
    file.close()
    return data_list


def create_vector_output(data_list):
    data_vectors = np.zeros((len(data_list), emb_dim))
    label_vectors = np.zeros((len(data_list), emb_dim))
    for i in range(len(data_list)):
        (hypo, hyper) = data_list[i]
        data_vectors[i] = word_vectors[hypo]
        label_vectors[i] = word_vectors[hyper]
    return data_vectors,label_vectors


def define_supervised_proj_network():
    supervised_input = Input(shape=(emb_dim,), name='supervised_input')
    supervised_hidden = Dense(emb_dim, activation='tanh', name='supervised_hidden')(supervised_input)
    pos_supervised_output = Dense(emb_dim, activation='linear', name='pos_supervised_output')(supervised_hidden)
    neg_supervised_output = Dense(emb_dim, activation='linear', name='neg_supervised_output')(supervised_hidden)
    model = Model(inputs=[supervised_input], outputs=[pos_supervised_output, neg_supervised_output])
    return model


def define_probase_proj_network():
    probase_input = Input(shape=(emb_dim,), name='probase_input')
    probase_hidden = Dense(emb_dim, activation='tanh', name='probase_hidden')(probase_input)
    pos_supervised_output = Dense(emb_dim, activation='linear', name='pos_probase_output')(probase_hidden)
    neg_supervised_output = Dense(emb_dim, activation='linear', name='neg_probase_output')(probase_hidden)
    model = Model(inputs=[probase_input], outputs=[pos_supervised_output, neg_supervised_output])
    return model


def define_positive_adverse_classifier():
    input = Input(shape=(emb_dim,), name='input')
    model_positive_output = Input(shape=(emb_dim,), name='model_positive_output')
    concat = Concatenate()([input, model_positive_output])
    pos_classifier = Dense(1, activation='softmax', name='pos_classifier')(concat)
    model = Model(inputs=[input, model_positive_output], outputs=[pos_classifier])
    return model


def define_negative_adverse_classifier():
    input = Input(shape=(emb_dim,), name='input')
    model_negative_output = Input(shape=(emb_dim,), name='model_negative_output')
    concat = Concatenate()([input, model_negative_output])
    neg_classifier = Dense(1, activation='softmax', name='pos_classifier')(concat)
    model = Model(inputs=[input, model_negative_output], outputs=[neg_classifier])
    return model


def train_supervised_proj_network(model, pos_data, pos_labels, neg_data, neg_labels):
    #one pass training of the supervised projection sub-network
    for i in range(2):
        model.compile(optimizer='adam', loss='mse',
                      loss_weights=[1, 0])
        model.fit([pos_data], [pos_labels, pos_labels],
                  epochs=20, batch_size=128)
        model.compile(optimizer='adam', loss='mse',
                      loss_weights=[0, 1])
        model.fit([neg_data], [neg_labels, neg_labels],
                  epochs=20, batch_size=128)


def train_probase_proj_network(model, pos_data, pos_labels, neg_data, neg_labels):
    # one pass training of the probase projection sub-network
    for i in range(2):
        model.compile(optimizer='adam', loss='mse',
                        loss_weights=[1, 0])
        model.fit([pos_data], [pos_labels, pos_labels],
                    epochs=20, batch_size=128)
        model.compile(optimizer='adam', loss='mse',
                        loss_weights=[0, 1])
        model.fit([neg_data], [neg_labels, neg_labels],
                    epochs=20, batch_size=128)


def train_positive_adverse_network(model, input_data, supervised_positive_output, probase_positive_output):
    for i in range(2):
        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.fit([input_data, supervised_positive_output],[np.ones(shape=(len(input_data), 1))],epochs=10, batch_size=128)
        model.fit([input_data, probase_positive_output],[np.zeros(shape=(len(input_data), 1))],epochs=10, batch_size=128)


def train_negative_adverse_network(model, input_data, supervised_negative_output, probase_negative_output):
    for i in range(2):
        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.fit([input_data, supervised_negative_output], [np.ones(shape=(len(input_data), 1))], epochs=10, batch_size=128)
        model.fit([input_data, probase_negative_output], [np.zeros(shape=(len(input_data), 1))], epochs=10, batch_size=128)


# load glove
global emb_dim
emb_dim = 100
gensim_file = 'word_vectors_as_teal.txt'
word_vectors = gensim.models.KeyedVectors.load_word2vec_format(gensim_file, binary=False)  # GloVe Model
print('load model successfully')

pos_train_list, neg_train_list=load_training_set('train.txt')
pos_train_data_vectors, pos_train_label_vectors=create_vector_output(pos_train_list)
neg_train_data_vectors, neg_train_label_vectors=create_vector_output(neg_train_list)

pos_probase_list=load_probase_training_set('as_probase_pos.txt')
neg_probase_list=load_probase_training_set('as_probase_neg.txt')
pos_probase_data_vectors, pos_probase_label_vectors=create_vector_output(pos_train_list)
neg_probase_data_vectors, neg_probase_label_vectors=create_vector_output(neg_train_list)

supervised_proj_network=define_supervised_proj_network()
probase_proj_network=define_probase_proj_network()
positive_adverse_classifier=define_positive_adverse_classifier()
negative_adverse_classifier=define_negative_adverse_classifier()

#initialized training
train_supervised_proj_network(supervised_proj_network, pos_train_data_vectors, pos_train_label_vectors, neg_train_data_vectors, neg_train_label_vectors)
train_probase_proj_network(probase_proj_network, pos_probase_data_vectors, pos_probase_label_vectors, neg_probase_data_vectors, neg_probase_label_vectors)

#iterative training
iter=5 #number of iterations
for i in range(0,iter):
    # for positive data
    [pos_train_label_vectors_predict1, _]=supervised_proj_network.predict(pos_train_data_vectors)
    [pos_train_label_vectors_predict2, _]=probase_proj_network.predict(pos_train_data_vectors)
    train_positive_adverse_network(positive_adverse_classifier, pos_train_data_vectors, pos_train_label_vectors_predict1, pos_train_label_vectors_predict2)

    [_, neg_train_label_vectors_predict1] = supervised_proj_network.predict(neg_train_data_vectors)
    [_, neg_train_label_vectors_predict2] = probase_proj_network.predict(neg_train_data_vectors)
    train_negative_adverse_network(negative_adverse_classifier, neg_train_data_vectors, neg_train_label_vectors_predict1, neg_train_label_vectors_predict2)

    train_supervised_proj_network(supervised_proj_network, pos_train_data_vectors, pos_train_label_vectors,
                                  neg_train_data_vectors, neg_train_label_vectors)
    train_probase_proj_network(probase_proj_network, pos_probase_data_vectors, pos_probase_label_vectors,
                               neg_probase_data_vectors, neg_probase_label_vectors)

supervised_proj_network.save('proj_model.h5')

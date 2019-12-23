import numpy as np
import tensorflow as tf
import gensim
import random


def load_training_set(path, sample_rate):
    # load training data
    pos_list = list()
    neg_list = list()
    file = open(path)
    while 1:
        line = file.readline()
        if not line:
            break
        if random.random() >= sample_rate:
            continue
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


def load_probase_training_set(path, sample_rate):
    data_list = list()
    file = open(path)
    while 1:
        line = file.readline()
        if not line:
            break
        if random.random() >= sample_rate:
            continue
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
    return data_vectors, label_vectors


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# load glove
global emb_dim
emb_dim = 100
gensim_file = 'word_vectors_as_teal.txt'
word_vectors = gensim.models.KeyedVectors.load_word2vec_format(gensim_file, binary=False)  # GloVe Model
print('load model successfully')

# define base network inputs
base_entity_embed_pos = tf.placeholder(tf.float32, shape=[None, emb_dim], name='base_entity_embed_pos')
base_entity_embed_neg = tf.placeholder(tf.float32, shape=[None, emb_dim], name='base_entity_embed_neg')

# define base network parameters
base_W_share = tf.Variable(xavier_init([emb_dim, emb_dim]), name='base_W_share')
base_B_share = tf.Variable(tf.zeros([emb_dim]), name='base_B_share')
base_W_hyper = tf.Variable(xavier_init([emb_dim, emb_dim]), name='base_W_hyper')
base_B_hyper = tf.Variable(tf.zeros([emb_dim]), name='base_B_hyper')
base_W_non = tf.Variable(xavier_init([emb_dim, emb_dim]), name='base_W_non')
base_B_non = tf.Variable(tf.zeros([emb_dim]), name='base_B_non')
theta_base = [base_W_share, base_B_share, base_W_hyper, base_B_hyper, base_W_non, base_B_non]


# define base positive network structure
def base_positive_projection(base_entity_embed_pos):
    base_H_share = tf.nn.relu(tf.matmul(base_entity_embed_pos, base_W_share) + base_B_share)
    base_O_hyper = tf.matmul(base_H_share, base_W_hyper) + base_B_hyper
    return base_O_hyper


# define base negative network structure
def base_negative_projection(base_entity_embed_neg):
    base_H_share = tf.nn.relu(tf.matmul(base_entity_embed_neg, base_W_share) + base_B_share)
    base_O_non = tf.matmul(base_H_share, base_W_hyper) + base_B_hyper
    return base_O_non


# predicted outputs
base_out_hyper = base_positive_projection(base_entity_embed_pos)
base_out_non = base_negative_projection(base_entity_embed_neg)

# true outputs
base_hyper_embed = tf.placeholder(tf.float32, shape=[None, emb_dim], name='base_hyper_embed')
base_non_embed = tf.placeholder(tf.float32, shape=[None, emb_dim], name='base_non_embed')

# define tax network inputs
tax_entity_embed_pos = tf.placeholder(tf.float32, shape=[None, emb_dim], name='tax_entity_embed_pos')
tax_entity_embed_neg = tf.placeholder(tf.float32, shape=[None, emb_dim], name='tax_entity_embed_neg')

# define tax network parameters
tax_W_share = tf.Variable(xavier_init([emb_dim, emb_dim]), name='tax_W_share')
tax_B_share = tf.Variable(tf.zeros([emb_dim]), name='tax_B_share')
tax_W_hyper = tf.Variable(xavier_init([emb_dim, emb_dim]), name='tax_W_hyper')
tax_B_hyper = tf.Variable(tf.zeros([emb_dim]), name='tax_B_hyper')
tax_W_non = tf.Variable(xavier_init([emb_dim, emb_dim]), name='tax_W_non')
tax_B_non = tf.Variable(tf.zeros([emb_dim]), name='tax_B_non')
theta_tax = [tax_W_share, tax_B_share, tax_W_hyper, tax_B_hyper, tax_W_non, tax_B_non]


# define tax pos network structure
def tax_positive_projection(tax_entity_embed_pos):
    tax_H_share = tf.nn.relu(tf.matmul(tax_entity_embed_pos, tax_W_share) + tax_B_share)
    tax_O_hyper = tf.matmul(tax_H_share, tax_W_hyper) + tax_B_hyper
    return tax_O_hyper


# define tax neg network structure
def tax_negative_projection(tax_entity_embed_neg):
    tax_H_share = tf.nn.relu(tf.matmul(tax_entity_embed_neg, tax_W_share) + tax_B_share)
    tax_O_non = tf.matmul(tax_H_share, tax_W_non) + tax_B_non
    return tax_O_non


# predicted outputs
tax_out_hyper = tax_positive_projection(tax_entity_embed_pos)
tax_out_non = tax_negative_projection(tax_entity_embed_neg)

# true outputs
tax_hyper_embed = tf.placeholder(tf.float32, shape=[None, emb_dim], name='tax_hyper_embed')
tax_non_embed = tf.placeholder(tf.float32, shape=[None, emb_dim], name='tax_non_embed')

# define classifier parameters
cls_W_hyper = tf.Variable(xavier_init([emb_dim * 2, emb_dim]), name='cls_W_hyper')
cls_B_hyper = tf.Variable(tf.zeros([emb_dim]), name='cls_B_hyper')


# define hyper classifier
def hyper_positive_classifier(base_entity_embed_pos, base_out_hyper):
    hyper_logit = tf.matmul(tf.concat([base_entity_embed_pos, base_out_hyper], axis=1), cls_W_hyper) + cls_B_hyper
    hyper_prob = tf.nn.sigmoid(hyper_logit)
    return hyper_prob, hyper_logit


def hyper_negative_classifier(base_entity_embed_pos, tax_out_hyper):
    hyper_logit = tf.matmul(tf.concat([base_entity_embed_pos, tax_out_hyper], axis=1), cls_W_hyper) + cls_B_hyper
    hyper_prob = tf.nn.sigmoid(hyper_logit)
    return hyper_prob, hyper_logit


cls_W_non = tf.Variable(xavier_init([emb_dim * 2, emb_dim]), name='cls_W_non')
cls_B_non = tf.Variable(tf.zeros([emb_dim]), name='cls_B_non')


# define non classifier
def non_positive_classifier(base_entity_embed, base_out_non):
    non_logit = tf.matmul(tf.concat([base_entity_embed, base_out_non], axis=1), cls_W_non) + cls_B_non
    non_prob = tf.nn.sigmoid(non_logit)
    return non_prob, non_logit


def non_negative_classifier(base_entity_embed, tax_out_non):
    non_logit = tf.matmul(tf.concat([base_entity_embed, tax_out_non], axis=1), cls_W_non) + cls_B_non
    non_prob = tf.nn.sigmoid(non_logit)
    return non_prob, non_logit


hyper_prob_pos, hyper_logit_pos = hyper_positive_classifier(base_entity_embed_pos, base_out_hyper)
hyper_prob_neg, hyper_logit_neg = hyper_negative_classifier(base_entity_embed_pos, tax_out_hyper)

ad_hyper_labels_pos = tf.placeholder(tf.float32, shape=[None, 1], name='ad_hyper_labels_pos')
ad_hyper_labels_neg = tf.placeholder(tf.float32, shape=[None, 1], name='ad_hyper_labels_neg')

non_prob_pos, non_logit_pos = non_positive_classifier(base_entity_embed_neg, base_out_non)
non_prob_neg, non_logit_neg = non_positive_classifier(base_entity_embed_neg, tax_out_non)

ad_non_labels_pos = tf.placeholder(tf.float32, shape=[None, 1], name='ad_non_labels_pos')
ad_non_labels_neg = tf.placeholder(tf.float32, shape=[None, 1], name='ad_non_labels_neg')

# define loss
base_loss = tf.reduce_mean(tf.square(base_out_hyper - base_hyper_embed)) + tf.reduce_mean(
    tf.square(base_out_non - base_non_embed))
tax_loss = tf.reduce_mean(tf.square(tax_out_hyper - tax_hyper_embed)) + tf.reduce_mean(
    tf.square(tax_out_non - tax_non_embed))

# define adversarial loss
ad_hyper_loss = tf.nn.softmax_cross_entropy_with_logits(labels=ad_hyper_labels_pos, logits=hyper_logit_pos) \
                + tf.nn.softmax_cross_entropy_with_logits(labels=ad_hyper_labels_neg, logits=hyper_logit_neg)

ad_non_loss = tf.nn.softmax_cross_entropy_with_logits(labels=ad_non_labels_pos, logits=non_logit_pos) \
              + tf.nn.softmax_cross_entropy_with_logits(labels=ad_non_labels_neg, logits=non_logit_neg)

# ad_hyper_solver = tf.train.AdamOptimizer().minimize(ad_hyper_loss,
#                                                    var_list=[theta_base, cls_W_hyper, cls_B_hyper])

# ad_non_loss = tf.nn.softmax_cross_entropy_with_logits(labels=ad_non_labels, logits=non_logit)


# define solvers
base_solver = tf.train.AdamOptimizer().minimize(base_loss)
tax_solver = tf.train.AdamOptimizer().minimize(tax_loss)
ad_hyper_solver = tf.train.AdamOptimizer().minimize(ad_hyper_loss)
ad_non_solver = tf.train.AdamOptimizer().minimize(ad_non_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(20):
    for k in range(10):
        sample_rate = 0.25
        pos_train_list, neg_train_list = load_training_set('train.txt', sample_rate)
        pos_train_data_vectors, pos_train_label_vectors = create_vector_output(pos_train_list)
        neg_train_data_vectors, neg_train_label_vectors = create_vector_output(neg_train_list)

        pos_probase_list = load_probase_training_set('as_probase_pos.txt', sample_rate)
        neg_probase_list = load_probase_training_set('as_probase_neg.txt', sample_rate)
        pos_probase_data_vectors, pos_probase_label_vectors = create_vector_output(pos_train_list)
        neg_probase_data_vectors, neg_probase_label_vectors = create_vector_output(neg_train_list)

        # train projection models
        _, base_loss_curr = sess.run([base_solver, base_loss],
                                     feed_dict={base_entity_embed_pos: pos_train_data_vectors,
                                                base_hyper_embed: pos_train_label_vectors,
                                                base_entity_embed_neg: neg_train_data_vectors,
                                                base_non_embed: neg_train_label_vectors
                                                })

        _, tax_loss_curr = sess.run([tax_solver, tax_loss],
                                    feed_dict={tax_entity_embed_pos: pos_probase_data_vectors,
                                               tax_hyper_embed: pos_probase_label_vectors,
                                               tax_entity_embed_neg: neg_probase_data_vectors,
                                               tax_non_embed: neg_probase_label_vectors
                                               })

        total_proj_loss_curr = base_loss_curr + tax_loss_curr
        print(total_proj_loss_curr)

    sample_rate = 0.5
    # generate data for prediction
    pos_train_list, neg_train_list = load_training_set('train.txt', sample_rate)
    pos_train_data_vectors, pos_train_label_vectors = create_vector_output(pos_train_list)

    base_hyper_predict = sess.run(base_out_hyper,
                                  feed_dict={base_entity_embed_pos: pos_train_data_vectors})
    tax_hyper_predict = sess.run(tax_out_hyper,
                                 feed_dict={tax_entity_embed_pos: pos_train_data_vectors})
    # labels
    positive_labels = np.ones([len(pos_train_data_vectors), 1])
    negative_labels = np.zeros([len(pos_train_data_vectors), 1])

    # train positive ad classifier
    _, pos_ad_loss_curr = sess.run([ad_hyper_solver, ad_hyper_loss],
                                   feed_dict={base_entity_embed_pos: pos_train_data_vectors,
                                              base_out_hyper: base_hyper_predict,
                                              tax_entity_embed_pos: pos_train_data_vectors,
                                              tax_out_hyper: tax_hyper_predict,
                                              ad_hyper_labels_pos: positive_labels,
                                              ad_hyper_labels_neg: negative_labels})

    neg_train_data_vectors, neg_train_label_vectors = create_vector_output(neg_train_list)

    base_non_predict = sess.run(base_out_non,
                                feed_dict={base_entity_embed_neg: neg_train_data_vectors})
    tax_non_predict = sess.run(tax_out_non,
                               feed_dict={tax_entity_embed_neg: neg_train_data_vectors})
    # labels
    positive_labels = np.ones([len(neg_train_data_vectors), 1])
    negative_labels = np.zeros([len(neg_train_data_vectors), 1])

    # train negative ad classifier
    _, neg_ad_loss_curr = sess.run([ad_non_solver, ad_non_loss],
                                   feed_dict={base_entity_embed_neg: neg_train_data_vectors,
                                              base_out_non: base_non_predict,
                                              tax_entity_embed_neg: neg_train_data_vectors,
                                              tax_out_non: tax_non_predict,
                                              ad_non_labels_pos: positive_labels,
                                              ad_non_labels_neg: negative_labels})

saver = tf.train.Saver()
saver.save(sess, "model/as-teal")

# The TEAL (Taxonomy Enhanced Adversarial Learning) Framework for Hypernymy Prediction 

### By Chengyu Wang (https://chywang.github.io)

**Introduction:** This software is the implementation of the Taxonomy Enhanced Adversarial Learning (TEAL) framework for hypernymy prediction. It includes three algorithms: i) the unsupervised measure U-TEAL for unsupervised hypernymy classification, ii) the supervised model S-TEAL for supervised hypernymy detection and iii) the adversarial supervised model AS-TEAL for supervised hypernymy detection with background taxonomies for distinct supervision.

**Paper:** Wang et al. Improving Hypernymy Prediction via Taxonomy Enhanced Adversarial Learning. AAAI 2019


**Frameworks**

+ U-TEAL: Unsupervised hypernymy measure, in the u-teal package

Inputs

1. word_vectors_u_teal.txt: The embeddings of all words. The start of the first line is the number of words and the dimensionality of word embeddings. After that, each line contains the words and its embeddings. All the values in a line are separated by a blank (' '). In practice, the embeddings can be learned by all deep neural language models.

> NOTE: Due to the large size of neural language models, we only upload the embedding vectors of words that are required to the test set. Please use your own neural language model instead, if you would like to try the algorithm over your datasets.

2. probase_pos.txt and probase_negative.txt: Positive and negative sampled generated from the Microsoft Concept Graph. The format of the file is "word1 \t word2" pairs.

3. test.txt: The path of the testing set. The format of the testing set is "word1 \t word2 \t label" triples.

Codes

1. u_teal_train.py: The script for training the projection neural network of U-TEAL.

2. u_teal_predict.py: The script for making predictions over the test set by U-TEAL.

> The formats of inputs and codes of S-TEAL and AS-TEAL algorithms are generally the same as those of U-TEAL. Hence, we do not elaborate in the following.

+ S-TEAL: Supervised hypernymy classifier, in the s-teal package

Inputs

1. word_vectors_s_teal.txt: The embeddings of all words. 

2. train.txt: The path of the training set.

3. test.txt: The path of the testing set. 

Codes

1. s_teal_proj_train.py: The script for training the projection neural network of S-TEAL.

2. s_teal_cls_train.py:  The script for training the SVM based hypernymy relation classifier of S-TEAL.

3. s_teal_cls_predict.py: The script for making predictions over the test set by S-TEAL.

**Dependencies**

The main Python packages that we use in this project include but are not limited to:

1. keras: 2.0.3
2. gensim: 2.0.0
3. numpy: 1.15.4
4. tensorflow: 1.12.0
5. scikit-learn: 0.18.1

The codes can run properly under the packages of other versions as well.

**More Notes on the Algorithms**

This software is the implementation of the AAAI 2019 paper. The codes have been slightly modified to make them easier for NLP researcher to re-use. Due to size limitation, we only provide part of the data related to our paper. However, it should be noted that all the data and resources are publicly available. Please refer to the links and references for details.


**Citation**

If you find this software useful for your research, please cite the following paper.

> @inproceedings{aaai2019,<br/>
&emsp;&emsp; author = {Chengyu Wang and Xiaofeng He and Aoying Zhou},<br/>
&emsp;&emsp; title = {Improving Hypernymy Prediction via Taxonomy Enhanced Adversarial Learning},<br/>
&emsp;&emsp; booktitle = {Proceedings of the 33rd AAAI Conference on Artificial Intelligence},<br/>
&emsp;&emsp; year = {2019}<br/>
}

More research works can be found here: https://chywang.github.io.




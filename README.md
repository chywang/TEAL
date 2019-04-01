# The TEAL (Taxonomy Enhanced Adversarial Learning) Framework for Hypernymy Prediction 

### By Chengyu Wang (https://chywang.github.io)

**Introducion:** This software is the implementation of the Taxonomy Enhanced Adversarial Learning (TEAL) framework for hypernymy prediction. It includes three algorithms: i) the unsupervised measure U-TEAL for unsupervised hypernymy classification, ii) the supervised model S-TEAL for supervised hypernymy detection and iii) the adversarial supervised model AS-TEAL for supervised hypernymy detection with background taxonomies for distinct supervision.

**Paper:** Wang et al. Improving Hypernymy Prediction via Taxonomy Enhanced Adversarial Learning. AAAI 2019


**Frameworks**

+ TransductLeaner: The main software entry-point, with five input arguments required.

1. w2vPath: The embeddings of all Chinese words in either the training set or the testing set. The start of each line of the file is the Chinese word, followed by the embedding vectors. All the values in a line are separated by a blank (' '). In practice, the embeddings can be learned by all deep neural language models.

> NOTE: Due to the large size of neural language models, we only upload the embedding vectors of words in the training and testing sets. Please use your own neural language model instead, if you would like to try the algorithm over your datasets.

2. trainPath: The path of the training set in the format of "word1 \t word2 \t label" triples. As for the label, 1 is for the hypernymy relation and 0 is for the non-hypernymy relation.

3. testPath: The path of the testing set. The format of the testing set is the same as that of the training set.

4. outputPath: The path of the output file, containing the model prediction scores of all the pairs in the testing set. The output of each pair is a real value in (-1,1). (Please refer to the paper for detailed explanation.)

5. dimension: The dimensionality of the embedding vectors.

> NOTE: The default values can be set as: "word_vectors.txt", "train.txt", "test.txt", "output.txt" and "50".

+ Eval: A simple evaluation script,  with three input arguments required. It outputs Precision, Recall and F1-score  as the evaluation scores. 

1. truthPath: The path of the testing set, with human-labeled results.

2. predictPath: The path of the model output file,.

3. thres: A threshold in (-1,1) for the model to assign relation labels to Chinese word pairs. (Please refer to the parameter 'θ' in the paper.)

> NOTE: The default values can be set as: "test.txt", "output.txt" and "0.1".

**Dependencies**

The main Python packages that we use in this project include but are not limited to:

1. keras: 2.0.3
2. gensim: 2.0.0
3. numpy: 1.15.4
4. tensorflow: 1.12.0

**More Notes on the Algorithms**

This software is the implementation of our AAAI 2019 paper. The codes have been slightly modified to make them for research to re-use. Due to size limitation, we only provide part of the data related to our paper. However, it should be noted that all the data and resources are publicly available. Please refer to the links and references for details.


**Citation**

If you find this software useful for your research, please cite the following paper.

> @inproceedings{aaai2019,<br/>
&emsp;&emsp; author = {Chengyu Wang and Xiaofeng He and Aoying Zhou},<br/>
&emsp;&emsp; title = {Improving Hypernymy Prediction via Taxonomy Enhanced Adversarial Learning},<br/>
&emsp;&emsp; booktitle = {Proceedings of the 33rd AAAI Conference on Artificial Intelligence},<br/>
&emsp;&emsp; year = {2019}<br/>
}

More research works can be found here: https://chywang.github.io.




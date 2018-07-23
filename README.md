## Commonsense Knowledge Base Reasoning 

This repository implements course project as a part of NLP-Lab offered in Intelligent Systems track at University of Bonn.

This project implements neural network based models to score Knowledge Base tuples.
Here main focus is on commonsense knowledge base. In standard KB each triple is a tuple of form $(subject,predicate,object)$ and subject,predicate and object are represented by a unique token. In contrast to standard KB here subject and object are an arbitrary phrases thus represented by set of tokens.

For further detail on methods, evaluation setup and results refer to 
[report.ipynb](https://github.com/MdAsifKhan/NLP-Project)

#Dataset
We use ConceptNet as a representation of commonsense knowledge. All data used in this experiment can be downloaded from: (http://ttic.uchicago.edu/~kgimpel/commonsense.html).


#Usage
The implementation is structured as follows.
1. ```model.py ```

Contains implementation of different neural networks for scoring a tuple. Currently we provide following models:
* Bilinear Averaging Model
* Bilinear LSTM Model
* ER-MLP Averaging Model
* ER-MLP LSTM Model

2. ```utils.py```
Implementation of preprocessing class, negative sampling and other basic utilities.
Main class and functions:
* class preprocess
Implements method to read arbitrary phrase ConceptNet triples and convert them to token representation for training neural network models. 
* function sample_negatives
Implements negative sampling strategy. Sampling is done by alternatively corrupting head and tail of a triple.
* class TripleDataset
Data class to support with pytorch batch loader.

3. ```evaluation.py```
Contains implementation of different evaluation metric. For this project we mainly use auc score. The above file also implements other metric: mean rank, mean reciprocal rank and accuracy.

4. ```pretrained_embedding.py```
The scoring of tuple is highly dependent on initial embeddings used for training. To help model to better capture commonsense knowledge of ConceptNet we use pretraining. We create training data by combining ConceptNet tuples with natural language sentences of Open Mind Common Sense. 

5. ```run_experiment.py```
main file to evaluate neural network model for commonsense knowledge base completion.

#Requirements
1. Pytorch
2. Python3+

#Comments & Feedback
For any comment or feedback please contact Mohammad Asif Khan via mail.
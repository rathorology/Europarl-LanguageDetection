# Language Detection on Europarl (European Parliament Dataset)

## Dataset

The Europarl parallel corpus is extracted from the proceedings of the European Parliament. It includes versions in 21 European languages: Romanic (French, Italian, Spanish, Portuguese, Romanian), Germanic (English, Dutch, German, Danish, Swedish), Slavik (Bulgarian, Czech, Polish, Slovak, Slovene), Finni-Ugric (Finnish, Hungarian, Estonian), Baltic (Latvian, Lithuanian), and Greek.

We will use this dataset for langauge detection of 21 languages used in dataset. It can thought of a multilingual text classification problem and we will use char level features for this task.

## Data Preprocessing

For each data unit, we have taken the following pre-processing steps:

 1. Remove Tags <> and () brackets content
 2. Split into multiple sentences using \n split
 
and then saved it to dataframe lang_df for further processing or modelling.

## Feature Extraction

As we are using this dataset for language detection, we will choose char-level features, since this is multilinugal and we don't need much local domain information like sublevel classification or category classification for text, char-level features will be suited much better and can be used to create a unified vocablury with less diversification.

We will use sklearn train_test_split to split into into training and test data for model validation, count_vectorizer with char analyzer for char level features(X) and label encoder for language type or (y). 

Count Vectorizer creates a char-level vocablury of the whole text data and then uses that to represent each sentence or unit of pre-processed data.

## Model-1 Sklearn Multinomial Naive Bayes

Multinomial Naive Bayes classifier is suitable for classification with discrete features, is well suited for text features. 

We will use the training features generated to fit to the model, no hypertuning.

## Run Pipeline

Using pre-processing funtions to pre-process data, and then generating features, training the model and testing it on test set.

## Keras Bi-RNN Classification

Using Bi-RNN with LSTM Cell for language detection.

### Model Information
    * 75x2 units LSTM cell bi-directional with concatination of outputs
    * Softmax Activation
    * Crossentropy Loss
    * Adam Optimizer
    * Mannual Batch Training
    
## Tensorflow Bi-RNN Classification

### Model Information
    * 75x2 units LSTM cell bi-directional with concatination of outputs
    * Softmax Activation
    * Crossentropy Loss
    * Adam Optimizer
    * Mannual Batch Training
    
### Model Params
    * learning_rate = 0.01
    * n_epoch = 10

### Layer Params
    * vocab_size = 322
    * num_classes = 21
    * hidden_dim = 75
    * timesteps = 1

## Evaluation

Evaluation of all the three models on Fellowship.ai custom dataset.

### Classification Accuracy

    * MNB(Sklearn) ACC:  0.9707093359216391 , 
    * Keras BI-RNN ACC:  0.9718167850969849
    * TF BI-RNN Accuracy: 0.9703284

### F1 Score

    * MNB(Sklearn) ACC:  0.9707093359216391 , 
    * Keras BI-RNN ACC:  0.9720171125440312
    * TF BI-RNN Accuracy: 0.9705131168450581

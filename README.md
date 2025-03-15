# Movie Review Sentiment Analysis using CNNs

This project implements Convolutional Neural Networks (CNNs) for sentiment analysis of movie reviews, using the Rotten Tomatoes Movie Review dataset.

## Project Description

Sentiment analysis, or opinion mining, is a Natural Language Processing (NLP) task that involves identifying and extracting subjective information from text. [cite: 2, 3, 4, 5, 6] This project explores the use of CNNs to classify movie reviews as either positive or negative. [cite: 6, 7]

While Recurrent Neural Networks (RNNs) are traditionally favored for sequential data like text, this project demonstrates the effectiveness of CNNs for sentiment analysis. [cite: 4, 5]

Two different CNN architectures were implemented:

* Shallow CNN with one convolutional layer. [cite: 27]
* Deeper CNN with multiple convolutional layers. [cite: 28]

## Dataset

The dataset used is the Rotten Tomatoes Movie Review dataset. [cite: 25, 26] It consists of 10662 movie reviews, with an equal number of positive and negative reviews (5331 positive and 5331 negative). [cite: 26]

## Code

The training and evaluation code is available in the `movie_sentiment_analysis.ipynb` Jupyter Notebook.

## Data Preprocessing

The following steps were used to preprocess the data:

1.  Tokenizing the sentences into words. [cite: 30, 31, 32]
2.  Removing punctuation and non-word tokens. [cite: 30, 31, 40, 41, 42]
3.  Building a look-up that maps an index to each word and its word embedding. [cite: 31, 52]
4.  Convert sentences into vectors of indices. [cite: 32]

### Tokenization

Tokenization is the process of breaking down a string of text into smaller units called tokens. [cite: 32, 33] In this project, sentences are tokenized into individual words. [cite: 30, 31, 32]

### Cleaning

Punctuation and non-word tokens (like smileys) are removed to clean the data. [cite: 30, 31, 40, 41, 42] Stopwords were initially considered for removal, but this step was skipped as it was found to negatively impact performance. [cite: 43, 44, 45]

### Turning sentences into indices

A vocabulary of all unique words in the dataset is created. [cite: 45, 46] Each word is then mapped to a unique index. [cite: 45, 46] Sentences are converted into sequences of these indices. [cite: 32] Unknown words encountered during testing are assigned a specific index. [cite: 47, 48, 49, 50, 51] Sentences are padded to ensure consistent length for matrix representation. [cite: 47, 48, 49, 50, 51]

### Word Embeddings

Pre-trained GloVe word embeddings (300-dimensional vectors trained on Wikipedia corpus) are used to represent words. [cite: 52, 53] Words in the vocabulary are mapped to their corresponding GloVe embeddings. [cite: 52, 53, 54, 55, 66, 67, 68]

## Model Architecture

Two CNN architectures were used:

### 1. Shallow CNN

* Embedding layer [cite: 57, 66, 67, 68]
* Convolutional layer with one filter size of 3 and 512 features [cite: 57, 58, 59]
* 1-max pooling layer [cite: 59, 60]
* Softmax layer for classification output [cite: 59, 60]
* Dropout regularization (probability = 0.5) [cite: 61, 65]
* ReLU activation function [cite: 61, 65]
* Dense layer with sigmoid activation function [cite: 61, 65]

### 2. Deep CNN

* Embedding layer [cite: 63, 65, 66, 67, 68]
* Convolutional layer with three filter sizes (3, 4, and 5) and 128 features for each size. [cite: 63, 64]
* 1D convolution [cite: 65]
* Max-pooling layer [cite: 65]
* Dropout regularization (probability = 0.5) [cite: 65]
* ReLU activation function [cite: 65]
* Dense layer with sigmoid activation function [cite: 65]

Both models use the Adam optimizer. [cite: 66]

## Experiments

The following variables were explored during training:

* Freezing or unfreezing the embedding layer. [cite: 69, 70]
* Using the full vocabulary or a reduced vocabulary. [cite: 69, 70, 76, 77, 78]

### Freezing the embedding layer

Freezing the embedding layer means that the pre-trained word embeddings are not updated during training. [cite: 70, 71] Unfreezing allows the embeddings to be fine-tuned, which can improve performance. [cite: 72, 73, 74, 75, 84, 85, 86, 87]

### Reducing Vocabulary

Reducing the vocabulary by removing less frequent words can help to reduce noise and improve performance. [cite: 76, 77, 78] In this project, the vocabulary was reduced to the 14,000 most common words. [cite: 78, 79]

## Results

| Model                     | Frozen Embeddings and Full Vocab | Trainable Embeddings and Full Vocab | Trainable Embeddings and Reduced Vocab |
| :------------------------ | :------------------------------: | :---------------------------------: | :------------------------------------: |
| Shallow CNN               |              75%               |              77.4%              |                 84.2%                 |
| Deep CNN                  |              77%               |               78.1%               |                 84.62%                |

The results indicate that:

* Fine-tuning word embeddings (trainable embeddings) generally improves performance. [cite: 84, 85, 86, 87]
* Reducing the vocabulary to the 14,000 most frequent words significantly increases accuracy. [cite: 88, 89, 90, 91]

The models converged quickly, often within 3 epochs. [cite: 92, 93, 94, 95] Early stopping was used to prevent overfitting. [cite: 92, 93, 94, 95]

## Interpreting Mistakes

The model makes mistakes on nuanced opinions, reviews that focus on aspects other than the movie's quality (e.g., actor controversies), and sentences with sarcasm. [cite: 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 117, 118, 119]

## Conclusions

CNNs can be effectively used for sentiment analysis. [cite: 112, 113] Fine-tuning word embeddings and reducing the vocabulary size can improve model performance. [cite: 84, 85, 86, 87, 88, 89, 90, 91, 114] The model still struggles with nuanced language, references, and sarcasm. [cite: 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 117, 118, 119]

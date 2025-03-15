# Movie Review Sentiment Analysis using CNNs

This project implements Convolutional Neural Networks (CNNs) for sentiment analysis of movie reviews, using the Rotten Tomatoes Movie Review dataset.

## Project Description

Sentiment analysis, or opinion mining, is a Natural Language Processing (NLP) task that involves identifying and extracting subjective information from text. This project explores the use of CNNs to classify movie reviews as either positive or negative.

While Recurrent Neural Networks (RNNs) are traditionally favored for sequential data like text, this project demonstrates the effectiveness of CNNs for sentiment analysis.

Two different CNN architectures were implemented:

* Shallow CNN with one convolutional layer.
* Deeper CNN with multiple convolutional layers.

## Dataset

The dataset used is the Rotten Tomatoes Movie Review dataset. It consists of 10662 movie reviews, with an equal number of positive and negative reviews (5331 positive and 5331 negative).

## Code

The training and evaluation code is available in the `movie_sentiment_analysis.ipynb` Jupyter Notebook.

## Data Preprocessing

The following steps were used to preprocess the data:

1.  Tokenizing the sentences into words.
2.  Removing punctuation and non-word tokens.
3.  Building a look-up that maps an index to each word and its word embedding.
4.  Convert sentences into vectors of indices.

### Tokenization

Tokenization is the process of breaking down a string of text into smaller units called tokens. In this project, sentences are tokenized into individual words.

### Cleaning

Punctuation and non-word tokens (like smileys) are removed to clean the data. Stopwords were initially considered for removal, but this step was skipped as it was found to negatively impact performance.

### Turning sentences into indices

A vocabulary of all unique words in the dataset is created. Each word is then mapped to a unique index. Sentences are converted into sequences of these indices. Unknown words encountered during testing are assigned a specific index. Sentences are padded to ensure consistent length for matrix representation.

### Word Embeddings

Pre-trained GloVe word embeddings (300-dimensional vectors trained on Wikipedia corpus) are used to represent words. Words in the vocabulary are mapped to their corresponding GloVe embeddings.

## Model Architecture

Two CNN architectures were used:

### 1. Shallow CNN

* Embedding layer
* Convolutional layer with one filter size of 3 and 512 features
* 1-max pooling layer
* Softmax layer for classification output
* Dropout regularization (probability = 0.5)
* ReLU activation function
* Dense layer with sigmoid activation function

### 2. Deep CNN

* Embedding layer
* Convolutional layer with three filter sizes (3, 4, and 5) and 128 features for each size.
* 1D convolution
* Max-pooling layer
* Dropout regularization (probability = 0.5)
* ReLU activation function
* Dense layer with sigmoid activation function

Both models use the Adam optimizer.

## Experiments

The following variables were explored during training:

* Freezing or unfreezing the embedding layer.
* Using the full vocabulary or a reduced vocabulary.

### Freezing the embedding layer

Freezing the embedding layer means that the pre-trained word embeddings are not updated during training. Unfreezing allows the embeddings to be fine-tuned, which can improve performance.

### Reducing Vocabulary

Reducing the vocabulary by removing less frequent words can help to reduce noise and improve performance. In this project, the vocabulary was reduced to the 14,000 most common words.

## Results

| Model                     | Frozen Embeddings and Full Vocab | Trainable Embeddings and Full Vocab | Trainable Embeddings and Reduced Vocab |
| :------------------------ | :------------------------------: | :---------------------------------: | :------------------------------------: |
| Shallow CNN               |              75%               |              77.4%              |                 84.2%                 |
| Deep CNN                  |              77%               |               78.1%               |                 84.62%                |

The results indicate that:

* Fine-tuning word embeddings (trainable embeddings) generally improves performance.
* Reducing the vocabulary to the 14,000 most frequent words significantly increases accuracy.

The models converged quickly, often within 3 epochs. Early stopping was used to prevent overfitting.

## Interpreting Mistakes

The model makes mistakes on nuanced opinions, reviews that focus on aspects other than the movie's quality (e.g., actor controversies), and sentences with sarcasm.

## Conclusions

CNNs can be effectively used for sentiment analysis. Fine-tuning word embeddings and reducing the vocabulary size can improve model performance. The model still struggles with nuanced language, references, and sarcasm.

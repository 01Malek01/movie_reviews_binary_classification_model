# IMDB Sentiment Classification (Binary)

## Project Overview

- This project trains a simple neural network to predict whether an IMDB movie review is positive or negative.
- It uses the built-in IMDB dataset with 50,000 labeled reviews (25,000 train, 25,000 test).
- Reviews are converted into fixed-length numeric vectors so a Dense neural network can learn from them.

## Architecture Breakdown

- Input: A 10,000-dimensional binary vector where each position indicates whether a word appears in the review.
- Hidden: Two Dense layers with 16 units and ReLU activation to learn useful patterns.
- Output: One Dense unit with sigmoid activation that outputs a probability of positive sentiment.

## Mathematical Intuition

- The model begins with random weights.
- During training, it predicts a probability for each review and compares it to the true label using binary cross-entropy loss.
- The RMSprop optimizer updates the weights to reduce that loss, which improves predictions over time.

## How to Run

- Install dependencies: `pip install tensorflow`
- Run the script: `python TextBinaryClassification.py`
- The first run will automatically download the IMDB dataset.

## Code Notes

- The IMDB dataset is preprocessed into sequences of word indices.
- The script keeps the top 10,000 most frequent words and uses a 10,000-sample validation split.
- `start_char`, `oov_char`, and `index_from` keep word index mapping consistent for decoding.
- The word index is reversed to map indices back to words, and `[START]`/`[OOV]` tokens are added.
- The first review is decoded and printed to verify the mapping.
- Sequences are vectorized into 10,000-dimensional binary vectors for a dense network.
- Training uses the partial training set and validates on the held-out set.
- The model is a simple feed-forward network with two ReLU layers and a sigmoid output.
- Loss is binary cross-entropy with RMSprop optimization and accuracy tracking.
- The trained model is evaluated on the test set and prints loss and accuracy.

# data shape in the first layer is : (25000,10000) because we have 25,000 reviews and each review is represented as a vector of 10,000 dimensions (one for each word in the vocabulary). The output shape of the first layer is (None, 16) because we have 16 units in the first hidden layer and the batch size is not fixed (None). The output shape of the second layer is also (None, 16) because it has the same number of units as the first hidden layer. The output shape of the final layer is (None, 1) because it has a single unit that outputs a probability value between 0 and 1 for binary classification.

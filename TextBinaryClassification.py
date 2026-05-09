import numpy as np 
from tensorflow import keras
from tensorflow.keras import layers, models  
from tensorflow.keras.datasets import imdb

#  limit to the top	10,000 most frequently occurring words in the training data
NUM_WORDS = 10000
#  reserve 10,000 samples for validation
VALIDATION_SIZE = 10000

start_char = 1
oov_char = 2 # out-of-vocabulary character will be used to replace words that were cut out because of the num_words argument or because they were not seen in the training set.
index_from = 3

# default settings from keras docs
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
	path="imdb.npz",
	num_words=NUM_WORDS,
	skip_top=0,
	maxlen=None,
	seed=113,
	start_char=start_char,
	oov_char=oov_char,
	index_from=index_from)

#  A dictionary mapping words to an integer index
word_index = keras.datasets.imdb.get_word_index()

#  A dictionary mapping integer indices to words
inverted_word_index = dict(
    (i + index_from, word) for (word, i) in word_index.items()
)

inverted_word_index[start_char] = "[START]"
inverted_word_index[oov_char] = "[OOV]"

#  decoding the review back to English words
decoded_sequence = " ".join(inverted_word_index[i] for i in train_data[0])

# print(decoded_sequence)


# helper to vectorize the data, we will use one-hot encoding to convert the lists of integers into a binary matrix representation. This is a common way to prepare text data for use with neural networks.
def vectorize_sequences(sequences, dimension=NUM_WORDS):
	results = np.zeros((len(sequences), dimension), dtype=np.float32)
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1.0
	return results


x_train = vectorize_sequences(train_data)
print("---------------------x train after vectorizing",x_train)
x_test = vectorize_sequences(test_data)
print("---------------------x test after vectorizing",x_test)

y_train = np.asarray(train_labels, dtype=np.float32)
y_test = np.asarray(test_labels, dtype=np.float32)

# split the training data into a training set and a validation set, we will use the first 10,000 samples as the validation set and the remaining samples as the training set. 
x_val = x_train[:VALIDATION_SIZE]
partial_x_train = x_train[VALIDATION_SIZE:]

y_val = y_train[:VALIDATION_SIZE]
partial_y_train = y_train[VALIDATION_SIZE:]



model = models.Sequential([ 
	layers.Flatten(input_shape=(NUM_WORDS,)),
	layers.Dense(16, activation="relu", input_shape=(NUM_WORDS,)),
	layers.Dense(16, activation="relu"),
	layers.Dense(1, activation="sigmoid"),
])

model.compile(
	optimizer="rmsprop",
	loss="binary_crossentropy",
	metrics=["accuracy"],
)

history = model.fit(
	partial_x_train,
	partial_y_train,
	epochs=5,
	batch_size=512,
	validation_data=(x_val, y_val),
)

results = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", results[0])
print("Test accuracy:", results[1])
print("Full results:", results)
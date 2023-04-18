import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

from ass1 import *

def example_with_embedding_and_neural_net(train_set, test_set):
    sentences1 = [train_set[i][0] for i in range(len(train_set))]
    sentences2 = [test_set[i][0] for i in range(len(test_set))]

    labels1 = [train_set[i][1] for i in range(len(train_set))]
    labels2 = [test_set[i][1] for i in range(len(test_set))]

    vocab_size = 5_000
    embedding_dim = 16
    max_length = 1_000
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 100_000

    training_sentences = sentences1[0:training_size]
    validation_sentences = sentences1[training_size:]
    training_labels = labels1[0:training_size]
    validation_labels = labels1[training_size:]
    testing_sentences = sentences2
    testing_labels = labels2


    training_padded = pad_sequences(training_sentences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    validation_padded = pad_sequences(validation_sentences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    testing_padded = pad_sequences(testing_sentences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # Need this block to get it to work with TensorFlow 2.x
    training_padded = np.array(training_padded)
    training_labels = np.array(training_labels)
    validation_padded = np.array(validation_padded)
    validation_labels = np.array(validation_labels)
    testing_padded = np.array(testing_padded)
    testing_labels = np.array(testing_labels)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()

    num_epochs = 30
    history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(validation_padded, validation_labels), verbose=2)
    model.evaluate(testing_padded, testing_labels)
    y_predicted = model.predict(testing_padded)
    tf.math.confusion_matrix(testing_labels, y_predicted)

if __name__ == "__main__":
    datasets = load_datasets()
    prompts, txts = extract_txts_and_prompts(datasets)
    prompts_and_txts_concat = concat_txts_and_prompts(prompts, txts)
    prompts_with_labels = add_labels(prompts)
    txts_with_labels = add_labels(txts)
    concat_with_labels = add_labels(prompts_and_txts_concat)

    # Training on the first domain, test on the second
    train_set = concat_with_labels[0]
    train_set.extend(concat_with_labels[1])
    shuffle_dataset(train_set)
    test_set = concat_with_labels[2]
    test_set.extend(concat_with_labels[3])
    shuffle_dataset(test_set)


    example_with_embedding_and_neural_net(train_set, test_set)

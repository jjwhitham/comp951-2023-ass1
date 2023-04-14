import json
from copy import deepcopy
import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_datasets():
    filenames = (
        "set1_human.json",
        "set1_machine.json",
        "set2_human.json",
        "set2_machine.json",
        # "test.json",
    )
    datasets = []
    for filename in filenames:
        with open(f"datasets/{filename}", "r") as f:
            datasets.append(json.load(f))
    return datasets


def extract_txts_and_prompts(datasets):
    # datasets = deepcopy(datasets)
    prompts_sets = [
        [instance["prompt"] for instance in dataset] for dataset in datasets
    ]
    txts_sets = [[instance["txt"] for instance in dataset] for dataset in datasets]
    return prompts_sets, txts_sets


def concat_txts_and_prompts(prompts_sets, txts_sets):
    # prompts_sets = deepcopy(prompts_sets)
    # txts_sets = deepcopy(txts_sets)

    prompts_and_txts_concat_sets = []
    for prompts, txts in zip(prompts_sets, txts_sets):
        prompts_and_txts_concat = []
        for prompt, txt in zip(prompts, txts):
            concat = prompt.copy()
            # Use reserved token `1` to separate prompt and txt
            concat.extend([1])
            concat.extend(txt.copy())
            prompts_and_txts_concat.append(concat)
        prompts_and_txts_concat_sets.append(prompts_and_txts_concat)

    # prompts_and_txts_concat_sets = [
    #     [
    #         # Use reserved token `1` to separate prompt and txt
    #         np.concatenate((np.concatenate((prompt, [1]))), txt)
    #         for prompt, txt in zip(prompts, txts)
    #     ]
    #     for prompts, txts in zip(prompts_sets, txts_sets)
    # ]
    return prompts_and_txts_concat_sets


def shuffle_dataset(dataset, seed=42):
    random.seed(seed)
    random.shuffle(dataset)


def add_labels(datasets):
    datasets_with_labels = []
    for i, dataset in enumerate(datasets):
        # 1 for human, 0 for machine
        label = (i + 1) % 2
        instance_with_label = []
        for instance in dataset:
            instance_with_label.append((instance, label))
        datasets_with_labels.append(instance_with_label)
    return datasets_with_labels


datasets = load_datasets()
prompts, txts = extract_txts_and_prompts(datasets)
prompts_and_txts_concat = concat_txts_and_prompts(prompts, txts)
prompts_with_labels = add_labels(prompts)
txts_with_labels = add_labels(txts)
concat_with_labels = add_labels(prompts_and_txts_concat)

# Train on the first domain, test on the second
train_set = concat_with_labels[0]
train_set.extend(concat_with_labels[1])
shuffle_dataset(train_set)
test_set = concat_with_labels[2]
test_set.extend(concat_with_labels[3])
shuffle_dataset(test_set)

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

example_with_embedding_and_neural_net(train_set, test_set)

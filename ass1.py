import json
from copy import deepcopy
import numpy as np
import random


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
import matplotlib.pyplot as plt
import statistics as st
from pprint import pprint

from dataloaders import *

VOCAB_SIZE = 5000

def get_dataset_stats(dataset):

    def flatten_dataset(dataset):
        flattened = []
        for instance in dataset:
            flattened.extend(instance.copy())
        return flattened

    def get_simple_stats(flattened_dataset):
        data_min = min(flattened_dataset)
        data_max = max(flattened_dataset)
        data_mode = st.mode(flattened_dataset)
        n_zeros = flattened_dataset.count(0)
        n_ones = flattened_dataset.count(1)
        frequencies = [flattened_dataset.count(i) for i in range(VOCAB_SIZE)]
        return data_min, data_max, data_mode, n_zeros, n_ones, frequencies

    def get_instance_lengths(dataset):
        return [len(instance) for instance in dataset]

    instance_lengths = get_instance_lengths(dataset)
    instance_lengths_stats = get_simple_stats(instance_lengths)
    flattened_dataset = flatten_dataset(dataset)
    simple_stats = get_simple_stats(flattened_dataset)

    return flattened_dataset, simple_stats, instance_lengths, instance_lengths_stats


def plot_histogram(flattened_dataset, bins=100):
    plt.style.use('ggplot')
    plt.hist(flattened_dataset, bins=bins)
    plt.show()

if __name__ == "__main__":
    datasets = load_datasets()
    txts, prompts = extract_txts_and_prompts(datasets)
    prompts_and_txts_concat = concat_txts_and_prompts(prompts, txts)
    set1_human_txts = txts[0]
    set1_machine_txts = txts[1]
    set2_human_txts = txts[2]
    set2_machine_txts = txts[3]

    set1_human_prompts = prompts[0]
    set1_machine_prompts = prompts[1]
    set2_human_prompts = prompts[2]
    set2_machine_prompts = prompts[3]

    set1_human_txts_stats = get_dataset_stats(set1_human_txts)
    plot_histogram(set1_human_txts_stats[0])

    _, simple_stats, _, instance_lengths_stats = set1_human_txts_stats

    # set1_machine_txts_stats = get_dataset_stats(set1_machine_txts)
    # plot_histogram(set1_machine_txts_stats[0])

    # set2_machine_txts_stats = get_dataset_stats(set2_machine_txts)
    # plot_histogram(set2_machine_txts_stats[0])

    # _, simple_stats, _, instance_lengths_stats = set1_machine_txts_stats

    pprint(simple_stats[:-1])
    pprint(instance_lengths_stats[:-1])
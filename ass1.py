from dataloaders import *

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

import context

import pickle

import torch

import training
from training import datasets, utils

if __name__ == '__main__':

    utils.set_random_state()
    tmg_gan = training.MYMETHOD()
    tmg_gan.fit(training.datasets.TrDataset())
    # count the max number of samples
    # max_cnt = max([len(tmg_gan.samples[i]) for i in tmg_gan.samples.keys()])
    # generate samples
    target_count = max([len(tmg_gan.samples[i]) for i in tmg_gan.samples.keys()])
    print(f"Target sample count for each class: {target_count}")
    for i in tmg_gan.samples.keys():
        current_count = len(tmg_gan.samples[i])
        cnt_generated = max(0, target_count - current_count)
        if cnt_generated > 0:
            generated_samples = tmg_gan.generate_qualified_samples(i, cnt_generated)
            generated_labels = torch.full([cnt_generated], i)
            datasets.tr_samples = torch.cat([datasets.tr_samples, generated_samples])
            datasets.tr_labels = torch.cat([datasets.tr_labels, generated_labels])
            print(f"Generated {cnt_generated} samples for class {i}")
        else:
            print(f"Class {i} already has {current_count} samples (>= target)")

    with open('dpl_data.pkl', 'wb') as f:
        pickle.dump(
            (
                datasets.tr_samples.numpy(),
                datasets.tr_labels.numpy(),
                datasets.te_samples.numpy(),
                datasets.te_labels.numpy(),
            ),
            f,
        )
        
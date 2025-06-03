from pathlib import Path

import datasets
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

print('Loading dataset...')
dataset = datasets.load_from_disk('')  # Path to dataset, fill in yourself
print('Dataset loaded')

lengths = []
for row in tqdm(dataset.select(range(1_000_000))):
    lengths.append(
        (row['fortify'], row['optimization_level'], len(row['asm_tokenized']))
    )

counts = pd.DataFrame(lengths, columns=['fortify', 'optimization_level', 'len'])

for fortify, fort_df in counts.groupby('fortify'):

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 10), sharex=True, sharey=True)
    axs = axs.flatten()
    for i, (opt, opt_df) in enumerate(fort_df.groupby('optimization_level')):
        counts = opt_df['len']
        fraction_larger_than_512 = sum(c > 512 for c in counts) / len(counts)
        print(f'Opt {opt}, fortify {fortify}: Percentage of sequences above 512 tokens: {fraction_larger_than_512:.2%}')

        counts = [min(c, 1024) for c in counts]
        binwidth = 10
        axs[i].hist(counts, bins=range(0, 1024 + binwidth, binwidth))

        axs[i].set_title(f'Opt {opt}: {fraction_larger_than_512:.2%} > 512')
        axs[i].set_xlabel('sequence length')
        axs[i].set_ylabel('count')

    plt.suptitle(f'ARM64 function lengths. Fortify = {fortify}')
    fig.tight_layout()

    Path("results").mkdir(parents=True, exist_ok=True)
    fort_string = 'fortify' if fortify else 'no-fortify'
    plt.savefig(f'results/sequence_lengths_ARM64_{fort_string}.png')

import json
from pathlib import Path
import pickle
import re
import sys

from datasets import Dataset


def parse_arm64_pickles(folder):
    pattern = re.compile(
        r"""
        ^
        (?P<bin_name>.+?)             # name of the binary
        -
        (?P<optimization_level>o.)    # optimization level expressed as O1, Os, ...
        -
        (?P<fortify>(?:no-)?fortify)  # optional fortification, either "fortify" or "no-fortify"
        -
        (?P<hash_value>[0-9a-f]+)     # hexadecimal hash value
        $
        """,
        re.IGNORECASE | re.VERBOSE
    )

    for path in Path(folder).rglob('**/*.pkl'):
        if not (match := pattern.match(path.stem)):
            raise ValueError(f'cannot recognize parts in "{path.stem}"')

        bin_name, optimization_level, fortify, hash_value = match.groups()

        with open(path, 'rb') as f:
            for func_name, func_data in pickle.load(f).items():
                _, _, _, cfg, _ = func_data
                # encode the CFG as JSON, arrow doesn't like complex data structures
                cfg = json.dumps([[block_id, block['asm']]
                                  for block_id, block in sorted(cfg.nodes.items())])
                if cfg != "[]":
                    yield {
                        'bin_name': bin_name,
                        'func_name': func_name,
                        'optimization_level': optimization_level,
                        'fortify': fortify == 'fortify',  # turn "fortify" or "no-fortify" into a boolean
                        'hash_value': hash_value,
                        'cfg': cfg,
                    }
                else:
                    print(f"{path} {func_name} contains an empy cfg")


if __name__ == '__main__':
    project_root = Path('') # Path to project root, fill in yourself
    source_folder = project_root / '' # Path to source folder from project root, fill in yourself
    target_folder = project_root / '' # Path to target folder from project root, fill in yourself

    print('Transforming data...')
    dataset = Dataset.from_generator(generator=parse_arm64_pickles,
                                     gen_kwargs={'folder': source_folder})
    dataset.save_to_disk(target_folder)

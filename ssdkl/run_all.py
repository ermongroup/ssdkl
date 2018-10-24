import os
import itertools

import yaml
import click

from ssdkl.run_experiment import main


alphas = [0.1, 1, 10]


def run_job(config):
    print(config)
    main(config)


def generate_configs(configs):
    # find the ones we need to loop over
    combo_items = {k: v for k, v in configs.items() if isinstance(v, list)}
    for curr_combo in itertools.product(*combo_items.values()):
        curr_combo_dict = {k: curr_combo[i]
                           for i, k in enumerate(combo_items.keys())}
        # make a copy of configs and update it
        configs_copy = configs.copy()
        configs_copy.update(curr_combo_dict)
        if configs_copy['model'] == 'semisup':
            # for semisup also loop over alphas
            for alpha in alphas:
                configs_copy_alpha = configs_copy.copy()
                configs_copy_alpha['alpha'] = alpha
                yield configs_copy_alpha
        else:
            yield configs_copy


@click.command()
@click.option('--config_file', default='config.yaml', required=True)
def run(config_file):
    # config
    config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), config_file)

    with open(config_path, 'r') as f:
        config_template = yaml.safe_load(f)

    for config in generate_configs(config_template):
        run_job(config)


if __name__ == "__main__":
    run()

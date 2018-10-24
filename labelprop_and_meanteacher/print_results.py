import os

import pandas as pd
import click


dss = ['skillcraft', 'parkinsons', 'elevators', 'protein', 'blog',
       'ctslice', 'buzz', 'electric']


@click.command()
@click.option('--results_dir', default='results_labelprop',
              help='Dataset where results are stored', required=True)
def print_results(results_dir):

    results_100 = pd.Series()
    results_300 = pd.Series()
    for ds in dss:
        try:
            results_path = os.path.join(results_dir, '{}.csv'.format(ds))

            df = pd.read_csv(results_path)

            results_100 = results_100.append(pd.Series({ds: df[df['n'] == 100]['test_loss']}))
            results_300 = results_300.append(pd.Series({ds: df[df['n'] == 300]['test_loss']}))
        except Exception:
            pass
    print("Num labeled = 100")
    print(results_100)
    print("Num labeled = 300")
    print(results_300)


if __name__ == "__main__":
    print_results()

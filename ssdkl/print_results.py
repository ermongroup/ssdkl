import os

import numpy as np
import yaml
import pandas as pd
import click


alphas = [0.1, 1, 10]


def get_rmse(curr_results_dir, model, config):
    if model != 'semisup':
        curr_results_dir = os.path.join(curr_results_dir, 'results')
    else:
        # choose best validation alpha
        mean_mse_vals = []
        mean_mse_val_dirs = []
        for alpha in alphas:
            alpha_results_dir = os.path.join(curr_results_dir + '_alpha{}'.format(alpha), 'results')
            # MSE shapes : (num_trials x num_iterations)
            mse_val = np.load(os.path.join(alpha_results_dir, 'mses_val.npy'))
            # choose best validation loss and get corresponding test
            mean_mse_vals.append(np.min(mse_val, axis=1))
            mean_mse_val_dirs.append(alpha_results_dir)
        mean_mse_vals = np.asarray(mean_mse_vals)
        curr_results_dir_list = [
            mean_mse_val_dirs[i] for i in np.argmin(mean_mse_vals, axis=0)]

    if model == 'coreg':
        mse_test = np.load(os.path.join(curr_results_dir, 'mses_test.npy'))
        last_rmse_tests = [np.sqrt(mse_test[i][~np.isinf(mse_test[i])][-1])
                          for i in range(mse_test.shape[0])]
        mean_rmse_test = np.mean(last_rmse_tests)
        std_rmse_test = np.std(last_rmse_tests)
    elif model == 'semisup':
        mean_rmse_tests = []
        # MSE shapes : (num_trials x num_iterations)
        for i, curr_results_dir in enumerate(curr_results_dir_list):
            mse_val = np.load(os.path.join(curr_results_dir, 'mses_val.npy'))
            mse_test = np.load(os.path.join(curr_results_dir, 'mses_test.npy'))
            # choose best validation loss and get corresponding test
            best_iteration = np.argmin(mse_val[i])
            mean_rmse_tests.append(np.sqrt(mse_test[i, best_iteration]))
        mean_rmse_test = np.mean(mean_rmse_tests)
        std_rmse_test = np.std(mean_rmse_tests)
    else:
        mse_test = np.load(os.path.join(curr_results_dir, 'mses_test.npy'))
        # MSE shapes : (num_trials x num_iterations)
        mse_val = np.load(os.path.join(curr_results_dir, 'mses_val.npy'))
        # choose best validation loss and get corresponding test
        best_iterations = np.argmin(mse_val, axis=1)
        mean_rmse_tests = np.sqrt(mse_test[range(mse_test.shape[0]), best_iterations])
        mean_rmse_test = np.mean(mean_rmse_tests)
        std_rmse_test = np.std(mean_rmse_tests)
    return mean_rmse_test, std_rmse_test


@click.command()
@click.option('--config_file', default='config.yaml', required=True)
def run(config_file):
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), config_file)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    results_dir = config['results_dir']

    dataset_names = list(os.listdir(results_dir))
    # assuming all the datasets have the same models
    model_names = list(os.listdir(os.path.join(results_dir, dataset_names[0])))
    num_train = config['num_train']
    if not isinstance(num_train, list):
        num_train = [num_train]

    result_df = pd.DataFrame()
    for dataset in dataset_names:
        for model in model_names:
            for num_examples in num_train:
                curr_results_dir = os.path.join(results_dir, dataset, model,
                                                'n{}_trials10'.format(num_examples))
                try:
                    mean_rmse_test, std = get_rmse(curr_results_dir, model, config)
                    result_row = {'dataset': dataset, 'model': model, 'num_train': num_examples, 'RMSE': mean_rmse_test, 'STD': std}
                    result_df = result_df.append(result_row, ignore_index=True)
                except Exception:
                    result_row = {'dataset': dataset, 'model': model, 'num_train': num_examples, 'RMSE': None, 'STD': None}
                    result_df = result_df.append(result_row, ignore_index=True)
                    print("Skipping {} {} {}".format(dataset, model, num_examples))

    # calculate the percent change from DKL
    dkl_results = result_df[result_df['model'] == 'dkl']
    for num_examples in num_train:
        print("Number of training examples: {}".format(num_examples))
        rmse_df = pd.DataFrame()
        change_df = pd.DataFrame()
        curr_dkl_results = dkl_results[dkl_results['num_train'] == num_examples]
        for dataset in dataset_names:
            # DKL RMSE
            curr_dkl_rmse = float(curr_dkl_results[curr_dkl_results['dataset'] == dataset]['RMSE'])
            change_row = {'dataset': dataset}
            rmse_row = {'dataset': dataset}
            for model in model_names:
                # filter down to the specific dataset and model
                model_results = result_df[result_df['model'] == model]
                curr_model_results = model_results[model_results['num_train'] == num_examples]
                curr_model_rmse = float(curr_model_results[curr_model_results['dataset'] == dataset]['RMSE'])
                rmse_row[model] = curr_model_rmse
                change_row[model] = ((curr_dkl_rmse - curr_model_rmse) / curr_dkl_rmse) * 100
            rmse_df = rmse_df.append(rmse_row, ignore_index=True)
            change_df = change_df.append(change_row, ignore_index=True)
        print("RMSE")
        rmse_df = rmse_df[['dataset'] + [c for c in rmse_df.columns if c != 'dataset']]
        print(rmse_df)
        print("Median")
        print(rmse_df.median(axis=0))
        print("Relative to DKL")
        change_df = change_df[['dataset'] + [c for c in change_df.columns if c != 'dataset']]
        print(change_df)
        print("Median")
        print(change_df.median(axis=0))


if __name__ == "__main__":
    run()

import os

import click
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import yaml

import ssdkl


dataset_names = ['skillcraft', 'parkinsons', 'elevators', 'protein', 'blog',
                 'ctslice', 'buzz', 'electric']

# read data dir from config file
ssdkl_root = os.path.dirname(ssdkl.__file__)
with open(os.path.join(ssdkl_root, 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)


def label_propagation_regression(X_l, y_l, X_u, X_val, y_val, sigma_2):
    # concatenate all the X's and y's
    X_all = np.concatenate([X_l, X_u, X_val], axis=0)

    print("KNN init")
    knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
    knn.fit(X_l, y_l)

    # y_u init
    # y_u = np.zeros((X_u.shape[0], ))
    y_u = knn.predict(X_u)

    # y_val_pred init
    # y_val_pred = np.zeros((X_val.shape[0], ))
    y_val_pred = knn.predict(X_val)

    y_all = np.concatenate([y_l, y_u, y_val_pred])

    # compute the kernel
    print("Compute kernel")
    T = np.exp(-cdist(X_all, X_all, 'sqeuclidean') / sigma_2)
    # row normalize the kernel
    T /= np.sum(T, axis=1)[:, np.newaxis]

    print("kernel done")
    delta = np.inf
    tol = 5e-6
    i = 0
    while delta > tol:
        y_all_new = T.dot(y_all)
        # clamp the labels known
        y_all_new[:X_l.shape[0]] = y_l
        delta = np.mean(y_all_new - y_all)
        y_all = y_all_new
        i += 1
        val_loss = np.sqrt(np.mean(np.square(y_all[-X_val.shape[0]:] - y_val)))
        if i % 10 == 0:
            print("Iter {}: delta={}, val_loss={}".format(i, delta, val_loss))
        if i > 500:
            break

    # return final val loss
    return val_loss


def run_dataset(dataset):
    dataset_dir = os.path.join(config['data_dir'], dataset)
    results_dir = 'results_labelprop'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_path = os.path.join(results_dir, "{}.csv".format(dataset))
    fold_results_path = os.path.join(results_dir, "{}_folds.csv".format(dataset))

    results = pd.DataFrame()
    fold_results = pd.DataFrame()

    # for labeled data sizes 100 and 300
    for labeled_size in [100, 300]:
        test_losses = []
        best_val_losses = []
        # over 10 folds
        X = np.load(os.path.join(dataset_dir, "X.npy"))
        y = np.load(os.path.join(dataset_dir, "y.npy"))
        for fold in range(10):
            shuffle_idxs = np.arange(X.shape[0])
            np.random.shuffle(shuffle_idxs)
            X = X[shuffle_idxs]
            y = y[shuffle_idxs]

            # split data
            train_labeled_size = int(labeled_size*0.9)
            val_size = labeled_size - train_labeled_size
            X_labeled = X[:train_labeled_size]
            y_labeled = y[:train_labeled_size]
            X_val = X[train_labeled_size:train_labeled_size+val_size]
            y_val = y[train_labeled_size:train_labeled_size+val_size]

            X_unlabeled = X[labeled_size+1000:labeled_size+21000]

            # search over sigma_2
            sigma_2s = np.linspace(0.8, 3.0, 5)
            val_losses = Parallel(n_jobs=5)(delayed(
                label_propagation_regression)(X_labeled, y_labeled, X_unlabeled, X_val, y_val, sigma_2)
                for sigma_2 in sigma_2s)
            best_idx = np.argmin(val_losses)
            best_val_loss = val_losses[best_idx]
            best_sigma_2 = sigma_2s[best_idx]
            fold_result_row = {'dataset': dataset, 'n': labeled_size,
                               'fold': fold, 'best_val_loss': best_val_loss,
                               'best_sigma_2': best_sigma_2,
                               'sigma_2s': sigma_2s,
                               'val_losses': val_losses}
            fold_results = fold_results.append(fold_result_row, ignore_index=True)

            X_test = X[labeled_size: labeled_size+1000]
            y_test = y[labeled_size: labeled_size+1000]

            # test with the best
            test_loss = label_propagation_regression(X_labeled, y_labeled, X_unlabeled, X_test, y_test, best_sigma_2)
            test_losses.append(test_loss)
            best_val_losses.append(best_val_loss)
        mean_test_loss = np.mean(test_losses)
        mean_best_val_loss = np.mean(best_val_losses)
        print("{}, n={}, test loss={}, mean best_val_loss={}".format(dataset, labeled_size, mean_test_loss, mean_best_val_loss))
        result_row = {'dataset': dataset, 'n': labeled_size, 'test_loss': mean_test_loss, 'mean_val_loss': mean_best_val_loss}
        results = results.append(result_row, ignore_index=True)

    results.to_csv(results_path)
    fold_results.to_csv(fold_results_path)


@click.command()
@click.option('--dataset', help='Dataset name', required=True)
def run(dataset):
    if dataset == 'all':
        for ds in dataset_names:
            run_dataset(ds)
    else:
        run_dataset(dataset)


if __name__ == "__main__":
    run()

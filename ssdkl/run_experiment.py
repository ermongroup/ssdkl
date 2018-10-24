import sys
import time
import argparse
from yaml import safe_load, dump
import numpy as np
import os

import ssdkl.models.train_models as train_models
import ssdkl.trainers as trainers

#########################################################


def main(config):
    t_start = time.time()

    # Run experiment with config
    trainer = trainers.setup(config, config['use_timestamp'])
    trainer.run_trials()
    t_end = time.time()

    print 'Finished in {} seconds'.format(t_end - t_start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config', help='YAML config file for experiment')
    parser.add_argument(
        '--model', default=None, help='Model to use')
    parser.add_argument(
        '--dataset', default=None, help='Dataset to use')
    parser.add_argument(
        '--use_ranges', help='Uses parameter ranges', action='store_true')
    parser.add_argument(
        '--lr_gp', type=float, default=-1, help='GP learning rate')
    parser.add_argument(
        '--lr_nn', type=float, default=-1, help='NN learning rate')
    parser.add_argument(
        '--alpha', type=float, default=1, help='Unsupervised weighting')
    parser.add_argument(
        '--num_train', type=int, default=-1,
        help='Number of labeled training examples')
    parser.add_argument(
        '--max_iters', type=int, default=-1,
        help='Max number of iterations')
    parser.add_argument(
        '--num_trials', type=int, default=-1,
        help='Number of trials (different data splits)')
    parser.add_argument(
        '--objective', default='variance', help='Unsupervised objective')
    parser.add_argument(
        '--use_timestamp', default='no', help='Use timestamp when saving data')
    parser.add_argument(
        '--use_cnn', type=int, default=-1, help='Use CNN instead of NN if 1')
    args = parser.parse_args()
    main(args.config, vars(args))

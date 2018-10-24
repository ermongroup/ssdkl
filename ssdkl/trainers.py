import os
from datetime import datetime
from yaml import dump
import inspect
import importlib


def get_args(arglist, conf):
    """
    Get the argument values for arguments in an argument list
    from a configuration dictionary.
    """
    args = {}
    for argname in arglist:
        # do our best and hope that failures mean default values are present
        try:
            args[argname] = conf[argname]
        except Exception:
            pass
    return args


def get_timestamp(use_timestamp):
    """
    Returns the timestamp or an empty string if not using timestamps.
    """
    if not use_timestamp:
        return ''
    else:
        timestamp = datetime.now()
        timestamp = '_{}_{}_{}_{}_{}'.format(
            timestamp.month, timestamp.day, timestamp.hour,
            timestamp.minute, timestamp.second)
        return timestamp


def set_load_paths(config, config_key, model_dirname, dataset_results_dir,
                   trail_path='', trial_trail_path=''):
    trial_dir = 'n{}_trials{}{}'.format(
        config['num_train'],
        config['num_trials'],
        trial_trail_path)

    config[config_key] = os.path.join(
        dataset_results_dir,
        model_dirname,
        trial_dir, 'models', trail_path)


#########################################################


def setup(config, use_timestamp):

    if not config['final']:
        # Getting datetime of experiment
        timestamp = get_timestamp(use_timestamp)
        description = 'n{}_trials{}'.format(config['num_train'],
                                            config['num_trials'])

        if config['model'] == 'semisup':
            description += '_alpha{}'.format(config['alpha'])

        # Setting up model inputs
        data_dir = os.path.join(config['data_dir'], config['dataset'])
        dataset_results_dir = os.path.join(
            config['results_dir'], config['dataset'])

        if config['kernel'] == 'square_polynomial':
            load_from = '_sp'
        else:
            load_from = ''

        results_dir = os.path.join(
            dataset_results_dir,
            config['model'] + load_from,
            description)

        # Setting up DKL trainer
        if config['model'] == 'dkl':
            trainer_name = 'DKLTrainer'

        # Setting up Semisup and Transductive trainers
        elif config['model'] == 'semisup':
            trainer_name = 'SemisupDKLTrainer'

        elif config['model'] == 'semisup_withlocs':
            trainer_name = 'SemisupDKLWithLocTrainer'

        # Setting up VAT trainer
        elif config['model'] == 'vat':
            trainer_name = 'VATTrainer'

        # Setting up Coreg trainer
        elif config['model'] == 'coreg':
            config['max_iters'] = 10
            trainer_name = 'CoregTrainer'
        else:
            print 'Model misspecified'

        results_dir += timestamp
        config['data_dir'] = data_dir
        config['results_dir'] = results_dir

    # Dynamically loads trainers with config arguments
    TrainerClass = getattr(
        importlib.import_module('ssdkl.models.train_models'), trainer_name)
    arglist = inspect.getargspec(TrainerClass.__init__)[0]
    args = get_args(arglist[1:], config)
    trainer = TrainerClass(**args)

    # Save experiment config file
    config['final'] = True
    config['verbose'] = False
    with open(os.path.join(results_dir, 'config.yaml'), 'w') as file:
        file.write(dump(config))

    return trainer

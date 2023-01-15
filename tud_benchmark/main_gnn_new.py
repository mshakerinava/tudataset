import os
import io
import sys
import json
import random
import hashlib
import argparse
import builtins
import subprocess
import torch
import numpy as np
import auxiliarymethods.datasets as dp
from auxiliarymethods.gnn_evaluation import gnn_evaluation
from gnn_baselines.gnn_architectures import GraphConvNet


def hash_args(args_dict, no_hash):
    args_dict = {x: args_dict[x] for x in args_dict if x not in no_hash}
    args_str = json.dumps(args_dict, sort_keys=True, indent=4)
    args_hash = hashlib.md5(str.encode(args_str)).hexdigest()[:8]
    return args_hash


def main(**kwargs):
    kwargs['act_fn'] = getattr(torch.nn.functional, args.act_fn)
    max_num_epochs = kwargs['max_num_epochs']
    del kwargs['max_num_epochs']

    num_reps = 10

    ### Smaller datasets.
    dataset = [["IMDB-BINARY", False], ["IMDB-MULTI", False], ["NCI1", True], ["PROTEINS", True],
               ["REDDIT-BINARY", False], ["ENZYMES", True]]

    results = []
    for d, use_labels in dataset:
        # Download dataset.
        dp.get_dataset(d)

        # GraphConvNet, dataset d, layers in [1:6], hidden dimension in {32,64,128}.
        acc, s_1, s_2 = gnn_evaluation(
            GraphConvNet, d, [1, 2, 3, 4, 5], [32, 64, 128],
            max_num_epochs=max_num_epochs, batch_size=64, start_lr=0.01, num_repetitions=num_reps, all_std=True,
            **kwargs
        )
        print(d + " " + "GraphConvNet " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GraphConvNet " + str(acc) + " " + str(s_1) + " " + str(s_2))

    num_reps = 3
    print(num_reps)

    ### Midscale datasets.
    dataset = [["MOLT-4", True, True], ["Yeast", True, True], ["MCF-7", True, True]]

    for d, use_labels, _ in dataset:
        print(d)
        dp.get_dataset(d)

        # GraphConvNet, dataset d, 3 layers, hidden dimension in {64}.
        acc, s_1, s_2 = gnn_evaluation(
            GraphConvNet, d, [3], [64],
            max_num_epochs=max_num_epochs, batch_size=64, start_lr=0.01, num_repetitions=num_reps, all_std=True,
            **kwargs
        )
        print(d + " " + "GraphConvNet " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GraphConvNet " + str(acc) + " " + str(s_1) + " " + str(s_2))

    dataset = [["reddit_threads", False, False],
               ["github_stargazers", False, False],
               ]

    for d, use_labels, _ in dataset:
        print(d)
        dp.get_dataset(d)

        # GraphConvNet, dataset d, 3 layers, hidden dimension in {64}.
        acc, s_1, s_2 = gnn_evaluation(
            GraphConvNet, d, [3], [64], max_num_epochs=max_num_epochs,
            batch_size=64, start_lr=0.01, num_repetitions=num_reps, all_std=True,
            **kwargs
        )
        print(d + " " + "GraphConvNet " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GraphConvNet " + str(acc) + " " + str(s_1) + " " + str(s_2))

    for r in results:
        print(r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--conv-repeat', type=int, default=2)
    parser.add_argument('--max-num-epochs', type=int, default=200)
    parser.add_argument('--reduction', type=str, default='mean', choices=['sum', 'mean', 'max', 'min', 'none'])
    parser.add_argument('--act-fn', type=str, default='relu')
    parser.add_argument('--skip-connection', action='store_true', default=False)
    args = parser.parse_args()

    args_hash = hash_args(vars(args), no_hash=['seed', 'checkpoint_path', 'base_path'])

    FINISH_TEXT = '** finished successfully **'
    TAG = 'args-%s__seed-%02d' % (args_hash, args.seed)

    os.makedirs('logs', exist_ok=True)
    LOG_PATH = os.path.join('logs', TAG + '.txt')

    abort = False
    try:
        if subprocess.check_output(['tail', '-n', '1', LOG_PATH]).decode('utf-8').strip() == FINISH_TEXT:
            print('ABORT: experiment has already been performed')
            abort = True
    except:
        pass

    if abort:
        sys.exit(-1)

    LOG_STR = io.StringIO()
    LOG_FILE = open(LOG_PATH, 'w')

    old_print = print

    def new_print(*args, **kwargs):
        kwargs['flush'] = True
        old_print(*args, **kwargs)
        kwargs['file'] = LOG_STR
        old_print(*args, **kwargs)
        kwargs['file'] = LOG_FILE
        old_print(*args, **kwargs)

    builtins.print = new_print

    print('writing log to `%s`' % LOG_PATH)

    print('───────────── machine info ─────────────')
    print(subprocess.check_output(['uname', '-a']).decode('utf-8').strip())
    print(subprocess.check_output(['lscpu']).decode('utf-8').strip())
    print(subprocess.check_output(['nvidia-smi']).decode('utf-8').strip())
    print('────────────────────────────────────────')

    print(subprocess.check_output(['pip', 'freeze']).decode('utf-8').strip())
    kwargs = vars(args)
    print('args = %s' % json.dumps(kwargs, sort_keys=True, indent=4))


    def set_seed(seed):
        random.seed(seed, version=2)
        np.random.seed(random.randint(0, 2**32))
        torch.manual_seed(random.randint(0, 2**32))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    # ensure reproducibility (?)
    set_seed(args.seed)
    del kwargs['seed']

    main(**kwargs)

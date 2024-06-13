import argparse
from pathlib import Path
from disco_gp import DiscoGP

from pprint import pprint

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('task')

    parser.add_argument('-w', '--use-weight-masks', action='store_true')
    parser.add_argument('-e', '--use-edge-masks', action='store_true')
    parser.add_argument('-m', '--model-name', default='gpt2')
    parser.add_argument('-bs', '--batch-size', default=32, type=int)

    args = parser.parse_args()

    if args.task == 'ioi':
        task_type = 'ioi'
    elif args.task.startswith('P'):
        task_type = 'pararel'
    else:
        task_type = 'blimp'

    configs_path = Path('configs') / (task_type + '.yaml')
    disco_gp = DiscoGP(args, configs_path)

    # Full model performance
    result = disco_gp.evaluate()

    # Search for the circuits
    disco_gp.search_circuit()
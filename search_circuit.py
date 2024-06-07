import argparse
from pathlib import Path
from disco_gp import DiscoGP

from pprint import pprint

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('task')

    parser.add_argument('-w', '--prune-weights', action='store_true')
    parser.add_argument('-e', '--prune-edges', action='store_true')
    parser.add_argument('-m', '--model-name', default='gpt2')

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

    disco_gp.search_circuit()
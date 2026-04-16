import _init_paths
import matplotlib.pyplot as plt
import argparse
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='lasot')
parser.add_argument('--model_name', type=str, default='cycletrack')
parser.add_argument('--parameter_name', type=str, default='detection_EM')
args = parser.parse_args()

trackers.extend(trackerlist(name=args.model_name, parameter_name=args.parameter_name, dataset_name=args.dataset_name,
                            run_ids=None, display_name=args.parameter_name))

dataset = get_dataset(args.dataset_name)

print_results(trackers, dataset, args.dataset_name, merge_results=True, plot_types=('success', 'prec', 'norm_prec'),
              force_evaluation=True)


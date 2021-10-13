# import pydevd
# pydevd.settrace('10.4.×.×', port=11234, stdoutToServer=True, stderrToServer=True)
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
from rhf import RHF
from util import *
# import scipy.stats as stats
import matplotlib.colors as mcolors

color_map = {'kendall': 'green', 'weighted_kendall': 'red', 'rbo': 'skyblue', 'weighted_rbo': 'yellow',
             'Spearmanr': 'blue'}
colors = list(mcolors.TABLEAU_COLORS)


def run(datasets):
    datasets = sorted(datasets)
    print(datasets)

    num_datasets = len(datasets)
    num_iterations = 1
    num_trees = 100
    # measure for each tree in each rhf in each p
    # p = [x for x in np.arange(0.9, 1.0, 0.010)]
    p = [1.]
    print(p)
    num_p = len(p)

    # how many trees have better ap than rhf
    count = np.zeros((num_iterations, num_datasets))
    # the best tree in a rhf
    best_tree_ap = np.zeros((num_iterations, num_datasets))
    # ap array for each tree in each rhf
    ap_all = np.zeros((num_iterations, num_datasets, num_trees))
    measure_all = np.zeros((num_iterations, num_datasets, num_trees, num_p))

    for i in range(num_iterations):

        for j, dataset in enumerate(datasets):
            print('~~~~~~~~~~~~~~~~~~~~~~dataset: ', dataset)

            # dataset loaded
            dataset_path = os.path.join('./datasets/klf/', dataset)
            df = pd.read_csv(dataset_path, header=0)
            y = df["label"]
            num_examples = len(y)
            num_anomaly = y.value_counts()[1]
            X = df.drop("label", axis=1)
            X.columns = [int(x) for x in X.columns]

            # rhf scores
            black_box = RHF(num_trees=100, max_height=5, seed_state=-1, check_duplicates=True, decremental=False,
                            use_kurtosis=True)
            black_box.fit(X)
            rhf_scores, _ = black_box.get_scores()
            rhf_scores = rhf_scores / 100
            ap_rhf = average_precision_score(y, rhf_scores)

            # RHF for selections
            black_box = RHF(num_trees=num_trees, max_height=5, seed_state=-1, check_duplicates=True, decremental=False,
                            use_kurtosis=True)
            black_box.fit(X)
            _, scores_all = black_box.get_scores()

            for k, score_per_tree in enumerate(scores_all):
                ap_all[i][j][k] = average_precision_score(y, score_per_tree)

                # calculate measures
                for l, p_i in enumerate(p):
                    # measure_all[i][j][k][l] = rbo_tau(rhf_scores, score_per_tree, p=p_i, weight_rank=True)
                    measure_all[i][j][k][l] = kendall_tau(rhf_scores, score_per_tree, weighted=True)
            best_tree_ap[i][j] = np.max(ap_all[i][j])
            count[i][j] = np.sum(ap_all[i][j] > ap_rhf)

    # ap_all_mean = np.mean(ap_all, axis=0)
    for l, p_i in enumerate(p):
        # measure_current = np.mean(measure_all, axis=0)[:, k]
        measure_sorted = []
        ap_all_sorted = []
        for i in range(num_iterations):
            for j in range(num_datasets):
                measure_current = measure_all[i, j, :, l]
                indexes_sorted = np.argsort(measure_current)

                measure_sorted += list(sorted(measure_current))
                ap_all_sorted += list(ap_all[i][j][indexes_sorted])

        # plt.subplot(3, 2, k + 1)
        print(len(measure_sorted), len(ap_all_sorted))
        print(np.min(measure_all[0][0][:, 0]), ap_all[0][0][np.argmin(measure_all[0][0][:, 0])])
        print(measure_sorted[0], ap_all_sorted[0])
        plt.plot(measure_sorted, ap_all_sorted, 'o', color=colors[l], label='p_%.3f' % p_i)
        plt.legend(loc='upper right')

        plt.xlabel('%s' % 'weighted_kendall')
        plt.ylabel('ap')

    plt.savefig('./results/rbo_ap_p/%s.jpg' % 'weighted_kendall')
    plt.cla()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--datasets', required=True, nargs='+')
    # args = parser.parse_args()
    # datasets = args.datasets
    datasets = os.listdir('./datasets/klf')
    # datasets = ['penglobal']
    run(datasets)

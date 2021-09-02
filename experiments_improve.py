import pickle as pkl
import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from rhf import RHF
from util import *

attributes = dict()

def choose_from_forest(rhf_scores, score_per_tree_list, select_method=['weighted_kendall']):

    best_score_mean = np.zeros(len(score_per_tree_list[0]))
    best_ap = {}

    for i, method in enumerate(select_method):

        best_measure = -10000
        best_score = None
        for score_per_tree in score_per_tree_list:

            if method == 'weighted_kendall':
                measure_per_tree, _ = stats.weightedtau(rhf_scores, score_per_tree, rank=None)
            elif method == 'rbo':
                measure_per_tree = rbo_tau(rhf_scores, score_per_tree, weight_rank=False)
            elif method == 'weighted_rbo':
                measure_per_tree = rbo_tau(rhf_scores, score_per_tree, weight_rank=True)
            elif method == 'rbo_10':
                measure_per_tree = rbo_tau(rhf_scores, score_per_tree, sample_size=int(len(score_per_tree) * 0.1))
            elif method == 'rbo_5':
                measure_per_tree = rbo_tau(rhf_scores, score_per_tree, sample_size=int(len(score_per_tree) * 0.05))

            if measure_per_tree > best_measure:
                best_measure = measure_per_tree
                best_score = score_per_tree

        best_score_mean += best_score
        best_ap[method] = average_precision_score(y, best_score)

    best_score_mean /= len(select_method)
    if 'all' in select_method:
        best_ap['all'] = average_precision_score(y, best_score_mean)
    return best_ap


if __name__ == '__main__':
    # rhf model loaded
    datasets = ['penglobal']#, 'ionosphere', 'vowels_odds', 'mnist', 'musk']
    datasets = os.listdir('./datasets/klf')


    results = []
    datasets_saved = []
    select_method = ['rbo_5', 'weighted_kendall', 'all']
    for dataset in datasets:
        result = []
        print('~~~~~~~~~~~~~~~~~~~~~~dataset: ', dataset)
        best_ap = {}
        ap = np.zeros(10)
        for d in range(10):
            rhf_path = './results/models/many_trees/rhf_%s_%d.pkl' % (dataset, d)
            if not os.path.exists(rhf_path):
                continue
            rhf = pkl.load(open(rhf_path, 'rb'))

            # dataset loaded
            dataset_path = os.path.join('./datasets/klf/', dataset)
            df = pd.read_csv(dataset_path, header=0)
            y = df["label"]
            # sample_size = y.value_counts()[1]
            sample_size = int(df.shape[0] * 0.1)

            # rhf inference
            black_box = RHF(num_trees=100, max_height=5, split_criterion='kurtosis', dataset='penglobal')
            black_box.check_hash(df)
            scores = np.zeros(df.shape[0])


            score_per_tree_list = []
            for tree in rhf:
                score_per_tree = np.zeros(df.shape[0])

                if black_box.has_duplicates:
                    for leaf in tree.leaves:
                        samples_indexes = leaf.data_index
                        p = black_box.data_hash[samples_indexes].nunique() / black_box.uniques_
                        scores[samples_indexes] += np.log(1 / (p))
                        score_per_tree[samples_indexes] = np.log(1 / (p))

                else:
                    for leaf in tree.leaves:
                        samples_indexes = leaf.data_index
                        p = leaf.size / black_box.uniques_
                        scores[samples_indexes] += np.log(1 / (p))
                        score_per_tree[samples_indexes] = np.log(1 / (p))
                score_per_tree_list.append(score_per_tree)

            # compute ap for rhf
            rhf_scores = scores / len(rhf)
            ap[d] = average_precision_score(y, rhf_scores)


            # find the best tree in rhf
            best_ap_d = choose_from_forest(rhf_scores, score_per_tree_list, select_method=select_method)
            for method in select_method:
                if method not in best_ap.keys():
                    best_ap[method] = [best_ap_d[method]]
                else:
                    best_ap[method].append(best_ap_d[method])
        result.append(ap.mean())
        for method in select_method:
            result.append(np.mean(best_ap[method]))
        results.append(result)
        datasets_saved.append(dataset)


    results = np.array(results)
    results = results.T

    data = pd.DataFrame(results, columns=datasets_saved, index=['rhf_ap'] + select_method + ['all'])

    ap_all = data.index
    for ap_i in ap_all[1:]:
        data.loc[str(ap_i) + '_gap'] = data.loc['rhf_ap'] - data.loc[ap_i]
    for ap_i in ap_all[1:]:
        data.loc[ap_i + '_gap/%'] = data.loc[ap_i + '_gap'] / data.loc['rhf_ap'] * 100
        data.round({ap_i + '_gap/%': 4})

    data['mean'] = data.mean(axis=1)
    data['var'] = data.var(axis=1)
    data.to_csv('./results/statfile/data_many.csv')










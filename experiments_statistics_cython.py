import pickle as pkl
import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from rhf import RHF
from util import *
import argparse


def run(datasets):
    datasets = sorted(datasets)
    print(datasets)

    results = []
    datasets_saved = []
    for dataset in datasets:
        # if dataset not in ['kdd_ftp']:
        #    continue
        result = []
        print('~~~~~~~~~~~~~~~~~~~~~~dataset: ', dataset)

        # dataset loaded
        dataset_path = os.path.join('./datasets/klf/', dataset)
        df = pd.read_csv(dataset_path, header=0)
        y = df["label"]
        sample_size = y.value_counts()[1]
        X = df.drop("label", axis=1)
        X.columns = [int(x) for x in X.columns]
        # sample_size = int(df.shape[0] * 0.1)
        # result.append(sample_size)
        # results.append(result)
        # datasets_saved.append(dataset)
        #
        # continue
        num_iterations = 10
        ap = np.zeros(num_iterations)
        best_ap = {}
        count = np.zeros(num_iterations)
        best_tree_ap = np.zeros(num_iterations)
        num_trees = 500
        #data_index = []
        for d in range(num_iterations):

            black_box = RHF(num_trees=100, max_height=5, seed_state=-1, check_duplicates=True, decremental=False, use_kurtosis=True)
            black_box.fit(X)
            rhf_scores, _ = black_box.get_scores()
            rhf_scores = rhf_scores / 100
            # print(rhf_scores)

            ap[d] = average_precision_score(y, rhf_scores)
            print('%s: %.2f ' % (dataset, ap[d]))


            # RHF for selections
            black_box = RHF(num_trees=num_trees, max_height=5, seed_state=-1, check_duplicates=True, decremental=False, use_kurtosis=True)

            black_box.fit(X)
            _, scores_all = black_box.get_scores()
            for score_per_tree in scores_all:
                ap_per_tree = average_precision_score(y, score_per_tree)
                if ap_per_tree > ap[d]:
                    count[d] += 1
                if ap_per_tree > best_tree_ap[d]:
                    best_tree_ap[d] = ap_per_tree

            # calculate measures
            select_method = ['kendall', 'weighted_kendall', 'rbo', 'weighted_rbo', 'Spearmanr']
            measures = compute_measure(rhf_scores, scores_all, select_method=select_method)

            # x = range(0, num_trees)
            # for i in range(len(select_method)):
            scores_combination = []
            for method, measure in measures.items():
                # measure = measures[i]
                measure_sorted_index = np.argsort(-measure)
                # import ipdb
                # ipdb.set_trace()
                score_sorted_current = scores_all[measure_sorted_index[0]]
                ap_current = average_precision_score(y, score_sorted_current)
                if method in best_ap.keys():
                    best_ap[method].append(ap_current)
                else:
                    best_ap[method] = [ap_current]
                scores_combination.append(score_sorted_current)

            for i in range(len(select_method)):
                for j in range(i + 1, len(select_method)):
                    method_ij = select_method[i] + '_' + select_method[j]
                    combine_score_current = (scores_combination[i] + scores_combination[j]) / 2
                    ap_current = average_precision_score(y, combine_score_current)
                    if method_ij in best_ap.keys():
                        best_ap[method_ij].append(ap_current)
                    else:
                        best_ap[method_ij] = [ap_current]



        result.extend([df.shape[0], sample_size, ap.mean(), count.mean(), best_tree_ap.mean()])
        for method in best_ap.keys():
            result.append(np.mean(best_ap[method]))
        results.append(result)
        datasets_saved.append(dataset)


    results = np.array(results)
    results = results.T
    data_index = ['n', 'anomaly', 'rhf_ap', 'good_trees', 'best_tree'] + list(best_ap.keys())
    print(data_index)
    # import ipdb
    # ipdb.set_trace()

    data = pd.DataFrame(results, columns=datasets_saved, index=data_index)
    ap_all = data.index
    for ap_i in ap_all[4:]:
        data.loc[ap_i + '_gap'] = data.loc['rhf_ap'] - data.loc[ap_i]
    for ap_i in ap_all[4:]:
        data.loc[ap_i + '_gap/%'] = data.loc[ap_i + '_gap'] / data.loc['rhf_ap'] * 100
        data.round({ap_i + '_gap/%': 4})

    data['mean'] = data.mean(axis=1)
    data['var'] = data.var(axis=1)
    data.to_csv('./results/statfile/data_new_100.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--datasets', required=True, nargs='+')
    # args = parser.parse_args()
    # datasets = args.datasets
    datasets = os.listdir('./datasets/klf')
    run(datasets)









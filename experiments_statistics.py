import pickle as pkl
import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from rhf import RHF
from util import *


if __name__ == '__main__':
    # rhf model loaded
    datasets = ['penglobal', 'kdd_smtp', 'vowels_odds', 'mnist', 'musk']
    datasets = os.listdir('./datasets/klf')

    # format
    # rhf_ap,
    # ap_best_single_tree,
    # rbo_best_single_tree,
    # ap_surrogate,
    # rbo_surrogate,

    results = []
    datasets_saved = []
    for dataset in datasets:
        if dataset not in ['kdd99','http_logged','kdd_http','kdd_http_distinct','mulcross','cover','http_all']:
           continue
        result = []
        print('~~~~~~~~~~~~~~~~~~~~~~dataset: ', dataset)

        # dataset loaded
        dataset_path = os.path.join('./datasets/klf/', dataset)
        df = pd.read_csv(dataset_path, header=0)
        y = df["label"]
        sample_size = y.value_counts()[1]
        # sample_size = int(df.shape[0] * 0.1)
        # result.append(sample_size)
        # results.append(result)
        # datasets_saved.append(dataset)
        #
        # continue
        
        ap = np.zeros(10)
        best_ap = {}
        count = np.zeros(10)
        best_tree_ap = np.zeros(10)
        #data_index = []

        for d in range(10):
            rhf_path = './results/models/many_trees/rhf_%s_%d.pkl' % (dataset, d)
            if not os.path.exists(rhf_path):
                continue
            rhf = pkl.load(open(rhf_path, 'rb'))

            # rhf inference
            black_box = RHF(num_trees=100, max_height=5, split_criterion='kurtosis', dataset='penglobal')
            black_box.check_hash(df)
            scores = np.zeros(df.shape[0])

            for tree in rhf:

                if black_box.has_duplicates:
                    for leaf in tree.leaves:
                        samples_indexes = leaf.data_index
                        p = black_box.data_hash[samples_indexes].nunique() / black_box.uniques_
                        scores[samples_indexes] += np.log(1 / (p))

                else:
                    for leaf in tree.leaves:
                        samples_indexes = leaf.data_index
                        p = leaf.size / black_box.uniques_
                        scores[samples_indexes] += np.log(1 / (p))

            # compute ap for rhf
            rhf_scores = scores / len(rhf)
            ap[d] = average_precision_score(y, rhf_scores)

            rhf_path = './results/models/1000_trees/rhf_%s_%d.pkl' % (dataset, d)
            if not os.path.exists(rhf_path):
                continue
            rhf_1000 = pkl.load(open(rhf_path, 'rb'))
            score_per_tree_list = []
            for tree in rhf_1000:
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
                ap_per_tree = average_precision_score(y, score_per_tree)
                if ap_per_tree > ap[d]:
                    count[d] += 1
                if ap_per_tree > best_tree_ap[d]:
                    best_tree_ap[d] = ap_per_tree

            # find the best tree in rhf
            select_method = ['kendall', 'weighted_kendall', 'rbo', 'weighted_rbo']
            best_ap_d = choose_from_forest(rhf_scores, score_per_tree_list, y, select_method=select_method, sample_size=sample_size)
            for method in select_method:
                if method not in best_ap.keys():
                    best_ap[method] = [best_ap_d[method]]
                else:
                    best_ap[method].append(best_ap_d[method])
            select_method = ['weighted_rbo', 'weighted_kendall', 'all']
            best_ap_d = choose_from_forest(rhf_scores, score_per_tree_list, y, select_method=select_method, sample_size=sample_size)
            for method in ['weight_rbo_weight_kendall_all']:
                if method not in best_ap.keys():
                    best_ap[method] = [best_ap_d['all']]
                else:
                    best_ap[method].append(best_ap_d['all'])
            select_method = ['rbo', 'weighted_kendall', 'all']
            best_ap_d = choose_from_forest(rhf_scores, score_per_tree_list, y, select_method=select_method, sample_size=sample_size)
            for method in ['rbo_weight_kendall_all']:
                if method not in best_ap.keys():
                    best_ap[method] = [best_ap_d['all']]
                else:
                    best_ap[method].append(best_ap_d['all'])
            select_method = ['rbo', 'kendall', 'all']
            best_ap_d = choose_from_forest(rhf_scores, score_per_tree_list, y, select_method=select_method, sample_size=sample_size)
            for method in ['rbo_kendall_all']:
                if method not in best_ap.keys():
                    best_ap[method] = [best_ap_d['all']]
                else:
                    best_ap[method].append(best_ap_d['all'])


        result.extend([df.shape[0], sample_size, ap.mean(), count.mean(), best_tree_ap.mean()])
        for method in best_ap.keys():
            result.append(np.mean(best_ap[method]))

        # # surrogate model loaded
        # surrogate = STACISurrogatesKendalTauDis()  # 'R-squared')#'RBO')
        #
        # surrogate_path = './results/models/surrogate_%s.pkl' % dataset
        # surrogate.trees = pkl.load(open(surrogate_path, 'rb'))
        # surrogate_score = surrogate.predict_RHF(df)[0]
        # surrogate_ap = average_precision_score(y, surrogate_score)
        # # print('Ap for surrogate: ', surrogate_ap)
        #
        # result.extend([surrogate_ap])
        results.append(result)
        datasets_saved.append(dataset)


    results = np.array(results)
    results = results.T
    data_index = ['n', 'anomaly', 'rhf_ap', 'good_trees', 'best_tree'] + ['kendall', 'weighted_kendall', 'rbo', 'weighted_rbo', 'rbo_5', 'rbo_10', 'weight_rbo_weight_kendall_all', 'rbo_weight_kendall_all', 'rbo_kendall_all']
    #list(best_ap.keys()))
    print(data_index)
    import ipdb
    ipdb.set_trace()

    data = pd.DataFrame(results, columns=datasets_saved, index=data_index)
    ap_all = data.index
    for ap_i in ap_all[4:]:
        data.loc[ap_i + '_gap'] = data.loc['rhf_ap'] - data.loc[ap_i]
    for ap_i in ap_all[4:]:
        data.loc[ap_i + '_gap/%'] = data.loc[ap_i + '_gap'] / data.loc['rhf_ap'] * 100
        data.round({ap_i + '_gap/%': 4})

    data['mean'] = data.mean(axis=1)
    data['var'] = data.var(axis=1)
    data.to_csv('./results/statfile/data.csv')

    # data = pd.DataFrame(results, columns=datasets_saved, index=['anomaly'])
    # data.to_csv('./results/statfile/data_tmp.csv')









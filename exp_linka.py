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

        # EXAMPLE
        # centers = [[1, 1], [-1, -1], [1, -1]]    # 定义 3 个中心点
        # # 生成 n=750 个样本，每个样本的特征个数为 d=2，并返回每个样本的真实类别
        # X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)
        # import ipdb
        # ipdb.set_trace()
        # plt.figure(figsize=(10, 8))
        # plt.scatter(X[:, 0], X[:, 1], c='b')
        # plt.title('The dataset')
        # # plt.savefig('./results/linka/test.jpg')
        # Z = linkage(X, method='single', metric='euclidean')  # ward
        # plt.figure(figsize=(10, 8))
        # labels = fcluster(Z, t=15, criterion='distance')
        # labels = fcluster(Z, t=3, criterion='maxclust')
        # plt.figure(figsize=(10, 8))
        # plt.title('The Result of the Agglomerative Clustering')
        # plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='prism')
        # plt.savefig('./results/linka/test.jpg')
        # data_index = []
        num_iterations = 1
        num_cluster = 3
        ap = np.zeros(num_iterations)
        # count = np.zeros(num_iterations)
        # best_tree_ap = np.zeros(num_iterations)
        cluster_ap = {}
        average_ap = {}

        gap_num_cluster = []
        for d in range(num_iterations):
        # for d in range(num_cluster):
            black_box = RHF(num_trees=100, max_height=5, seed_state=-1, check_duplicates=True, decremental=False,
                            use_kurtosis=True)
            black_box.fit(X)
            rhf_scores, scores_list = black_box.get_scores()
            rhf_scores = rhf_scores / 100
            ap[d] = average_precision_score(y, rhf_scores)


            # np.all(np.isfinite(X))
            # Z = linkage(scores_list,  method='single', metric='correlation')                  #ward
            Z = linkage(scores_list,  method='complete', metric=rbo_tau)                  #ward

            # labels_pred = fcluster(Z, t=15, criterion='distance')
            labels_pred = fcluster(Z, t=num_cluster, criterion='maxclust')
            print(labels_pred)
            if len(np.unique(labels_pred)) == 1:
                continue

            # labels
            labels = np.zeros(len(labels_pred))
            ap_single_list = list(map(lambda x: average_precision_score(y, x), scores_list))
            ap_single_list = np.array(ap_single_list)
            ap_single_order = np.argsort(-ap_single_list)
            labels[ap_single_order[:int(len(labels_pred) / num_cluster)]] = 1
            labels[ap_single_order[int(len(labels_pred) / num_cluster):2 * int(len(labels_pred) / num_cluster)]] = 2
            labels[ap_single_order[2 * int(len(labels_pred) / num_cluster)]:] = 3

            print('Adjust mutual information: %.3f' %
                  adjusted_mutual_info_score(labels, labels_pred))

            select_method = ['rbo']#['kendall', 'weighted_kendall', 'rbo', 'weighted_rbo', 'Spearmanr']
            measures = compute_measure(rhf_scores, scores_list, select_method=select_method)

            for method, measure in measures.items():
                ap_iter = 0
                score_clusters = np.zeros(len(y))
                score_average = np.zeros(len(y))
                for cluster in range(num_cluster):
                    indexes, = np.where(labels_pred == cluster + 1)

                    measure_cluster = measure[indexes]
                    measure_sorted_index = np.argsort(-measure_cluster)

                    # import ipdb
                    # ipdb.set_trace()

                    score_sorted_current = scores_list[measure_sorted_index[0]]

                    score_clusters += score_sorted_current * len(indexes)
                    score_average += score_sorted_current

                score_clusters /= len(labels_pred)
                score_average /= num_cluster
                ap_clusters_d = average_precision_score(y, score_clusters)
                ap_average_d = average_precision_score(y, score_average)
                if method in cluster_ap.keys():
                    cluster_ap[method].append(ap_clusters_d)
                else:
                    cluster_ap[method] = [ap_clusters_d]
                if method in average_ap.keys():
                    average_ap[method].append(ap_average_d)
                else:
                    average_ap[method] = [ap_average_d]
                # print(cluster_ap, average_ap)

        result.extend([df.shape[0], sample_size, ap.mean()])
        for method in cluster_ap.keys():
            result.append(np.mean(cluster_ap[method]))
        for method in average_ap.keys():
            result.append(np.mean(average_ap[method]))
        print('rhf_ap: %.4f, cluster ap: %.4f, average_ap: %.4f' % (
        ap[d], np.mean(cluster_ap[method]), np.mean(average_ap[method])))

        results.append(result)
        datasets_saved.append(dataset)

    results = np.array(results)
    results = results.T
    data_index = ['n', 'anomaly', 'rhf_ap'] + [x + '_clu' for x in list(cluster_ap.keys())] + [x + '_ave' for x in list(average_ap.keys())]
    # print(data_index)
    #

    data = pd.DataFrame(results, columns=datasets_saved, index=data_index)
    ap_all = data.index
    for ap_i in ap_all[3:]:
        data.loc[ap_i + '_gap'] = data.loc['rhf_ap'] - data.loc[ap_i]
    for ap_i in ap_all[3:]:
        data.loc[ap_i + '_gap/%'] = data.loc[ap_i + '_gap'] / data.loc['rhf_ap'] * 100
        data.round({ap_i + '_gap/%': 4})

    data['mean'] = data.mean(axis=1)
    data['var'] = data.var(axis=1)
    data.to_csv('./results/statfile/data_linka.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', required=True, nargs='+')
    args = parser.parse_args()
    datasets = args.datasets
    datasets = os.listdir('./datasets/klf')
    run(datasets)


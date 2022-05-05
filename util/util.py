#coding=utf-8
import numpy as np
import rbo
import scipy.stats as stats
from sklearn.metrics import average_precision_score, mean_squared_error
import  time
import pandas as pd
# attributes = dict()

def traverse_rhf_tree(root, attributes):
    if root.attribute  is None:
        return
    if root.attribute not in attributes.keys():
        attributes[root.attribute] = root.depth
    else:
        attributes[root.attribute] +=root.depth

    if root.left is not None:
        traverse_rhf_tree(root.left,attributes)
    if root.right is not None:
        traverse_rhf_tree(root.right,attributes)

def kendall_tau(rhf_scores, score_per_tree, weighted=False):
    if not weighted:
        tau, p_value = stats.kendalltau(rhf_scores, score_per_tree)
    else:
        rhf_indices = np.argsort(-rhf_scores)
        rhf_ranking = np.argsort(rhf_indices)
        tau, p_value = stats.weightedtau(rhf_scores, score_per_tree, rank=rhf_ranking)

    # tau = (tau + 1) / 2.
    return tau

def weighted_kendall(rhf_scores, score_per_tree):
    rhf_indices = np.argsort(-rhf_scores)
    rhf_ranking = np.argsort(rhf_indices)
    tau, p_value = stats.weightedtau(rhf_scores, score_per_tree, rank=rhf_ranking)
    tau = (tau + 1) / 2.
    return tau

def weighted_kendall_modified(rhf_scores, score_per_tree):
    # measure smaller, similarity bigger
    rhf_indices = np.argsort(-rhf_scores)
    rhf_ranking = np.argsort(rhf_indices)
    tau, p_value = stats.weightedtau(rhf_scores, score_per_tree, rank=rhf_ranking)
    tau = 1 - (tau + 1) / 2.
    return tau

def rbo_tau(rhf_scores, score_per_tree, p=1.0, sample_size=None, weight_rank=False):
    # compute rbo for best tree
    # rhf_indices是：将rhf_scores从高到低排列时的元素索引
    # rhf_ranking是：rhf_scores每个元素的排序
    # 因此：rhf_ranking[rhf_indices] = [0, 1, 2, 3, ...,]
    rhf_indices = np.argsort(-rhf_scores)
    rhf_ranking = np.argsort(rhf_indices)

    per_tree_ranking = np.argsort(np.argsort(-score_per_tree))

    measure_time = time.time()

        # weighted_rank = True, weights = {1, 1/2, 1/3, 1/4.....}, measure is very small(0.0020 0.0007)
        # weighted_rank默认是false，即论文中定义的权重
        #     p=1时，weights=[1,1,1,1,.....]
        #     0<p<1时，weights=[1.0 * (1 - p) * p ** d for d in range(k)], k为深度

        # 两种计算rbo的传进去的排序不同，计算结果也不同。比如：（per_tree_rbo，per_tree_rbo_1） = 0.6037666787648835 0.4846807604101607
        # 因为rbo是越前面的元素重要性越高，因此按照从高到低的排序传进去，得到的measure会更准确些
        # rhf_indices_top = rhf_indices[:sample_size]
        # per_tree_rbo_1 = rbo.RankingSimilarity(rhf_ranking, per_tree_ranking).rbo(p=p, weight_rank=weight_rank)

    # print('ori: ', len(rhf_scores), ' sample: ', len(rhf_ranking[rhf_indices[:sample_size]]), ' sample_size: ', sample_size)
    per_tree_rbo = rbo.RankingSimilarity(rhf_ranking[rhf_indices[:sample_size]], per_tree_ranking[rhf_indices[:sample_size]]).rbo(p=p, weight_rank=weight_rank)
    per_tree_rbo_1 = rbo.RankingSimilarity(rhf_ranking[:sample_size], per_tree_ranking[:sample_size]).rbo(p=p, weight_rank=weight_rank)
    # print('before: ', rhf_ranking[rhf_indices[:sample_size]],' rbo: ', per_tree_rbo )
    # print('after: ', rhf_ranking, ' rbo: ', per_tree_rbo_1)


    return per_tree_rbo_1

def squared_error(rhf_scores, score_per_tree, sample_size=None):
    if sample_size is None:
        tau = np.sum((score_per_tree - rhf_scores) ** 2)
    else:
        tau = np.sum((score_per_tree[:sample_size] - rhf_scores[:sample_size]) ** 2)
    return tau

def choose_from_forest(rhf_scores, score_per_tree_list, y, select_method=['weighted_kendall'], sample_size=None):

    best_score_mean = np.zeros(len(score_per_tree_list[0]))
    best_ap = {}
    best_ap['tree_id'] = []

    for i, method in enumerate(select_method):

        best_measure = -10000
        best_score = [0] * len(score_per_tree_list[0])
        best_tree_id = -1

        if method == 'all':
            break
        for tree_id, score_per_tree in enumerate(score_per_tree_list):
            if method == 'kendall':
                measure_per_tree, _ = stats.kendalltau(rhf_scores, score_per_tree)
            elif method == 'weighted_kendall':
                measure_per_tree, _ = stats.weightedtau(rhf_scores, score_per_tree, rank=None)
            elif method == 'rbo':
                measure_per_tree = rbo_tau(rhf_scores, score_per_tree, weight_rank=False)
            elif method == 'weighted_rbo':
                measure_per_tree = rbo_tau(rhf_scores, score_per_tree, weight_rank=True)
            elif method == 'rbo_10':
                measure_per_tree = rbo_tau(rhf_scores, score_per_tree, sample_size=int(len(score_per_tree) * 0.1))
                # measure_per_tree = rbo_tau(rhf_scores, score_per_tree, sample_size=sample_size * 2)
            elif method == 'rbo_20':
                measure_per_tree = rbo_tau(rhf_scores, score_per_tree, sample_size=int(len(score_per_tree) * 0.2))
            elif method == 'rbo_5':
                measure_per_tree = rbo_tau(rhf_scores, score_per_tree, sample_size=int(len(score_per_tree) * 0.05))
                # measure_per_tree = rbo_tau(rhf_scores, score_per_tree, sample_size=sample_size)
            elif method == 'weighted_rbo_5':
                measure_per_tree = rbo_tau(rhf_scores, score_per_tree, weight_rank=True, sample_size=int(len(score_per_tree) * 0.05))
            elif method == 'weighted_rbo_10':
                measure_per_tree = rbo_tau(rhf_scores, score_per_tree, weight_rank=True, sample_size=int(len(score_per_tree) * 0.1))
            elif method == 'weighted_rbo_20':
                measure_per_tree = rbo_tau(rhf_scores, score_per_tree, weight_rank=True, sample_size=int(len(score_per_tree) * 0.2))

            if measure_per_tree > best_measure:
                best_measure = measure_per_tree
                best_score = score_per_tree
                best_tree_id = tree_id

        best_score_mean += best_score
        best_ap[method] = average_precision_score(y, best_score)
        best_ap['tree_id'].append(best_tree_id)

    best_score_mean /= len(select_method)
    if 'all' in select_method:
        best_ap['all'] = average_precision_score(y, best_score_mean)
    return best_ap

def compute_measure(rhf_scores, score_per_tree_list, method):
    if np.ndim(rhf_scores) == 1:
        rhf_scores = np.expand_dims(rhf_scores, axis=0)
    measures = np.zeros((len(rhf_scores), len(score_per_tree_list)))
    sample_size = len(rhf_scores[0])
    if len(rhf_scores[0]) > 40000:
        sample_size = int(0.02 * len(rhf_scores[0]))
    measure_time = time.time()
    for forest_id, rhf_score in enumerate(rhf_scores):

        for tree_id in range(forest_id, len(score_per_tree_list)):
            start_time = time.time()
            score_per_tree = score_per_tree_list[tree_id]

            if method == 'kendall':
                measure_per_tree, _ = stats.kendalltau(rhf_score, score_per_tree)
            elif method == 'weighted_kendall':
                measure_per_tree, _ = stats.weightedtau(rhf_score, score_per_tree, rank=None)
            elif method == 'rbo':
                measure_per_tree = rbo_tau(rhf_score, score_per_tree, p=1.0, sample_size=sample_size, weight_rank=False)
            elif method == 'weighted_rbo':
                measure_per_tree = rbo_tau(rhf_score, score_per_tree, weight_rank=True)
            elif method == 'rbo_10':
                measure_per_tree = rbo_tau(rhf_score, score_per_tree, sample_size=int(len(score_per_tree) * 0.1))
            elif method == 'rbo_5':
                measure_per_tree = rbo_tau(rhf_score, score_per_tree, sample_size=int(len(score_per_tree) * 0.05))
            elif method == 'rbo_20':
                measure_per_tree = rbo_tau(rhf_score, score_per_tree, sample_size=int(len(score_per_tree) * 0.2))
            elif method == 'weighted_rbo_5':
                measure_per_tree = rbo_tau(rhf_score, score_per_tree, weight_rank=True, sample_size=int(len(score_per_tree) * 0.05))
            elif method == 'weighted_rbo_10':
                measure_per_tree = rbo_tau(rhf_score, score_per_tree, weight_rank=True, sample_size=int(len(score_per_tree) * 0.1))
            elif method == 'Spearmanr':
                measure_per_tree, _ = stats.spearmanr(rhf_score, score_per_tree)
            elif method == 'mse':
                measure_per_tree = mean_squared_error(rhf_score[:sample_size], score_per_tree[:sample_size])

            # print('time for each iter: ', time.time()-start_time)
            measures[forest_id][tree_id] = measure_per_tree
    # print('measure time: ', time.time() - measure_time)

    return measures

def spearmanr(rhf_scores, score_per_tree):
    return stats.spearmanr(rhf_scores, score_per_tree)

def compute_measure_single(list1, list2, method):
    if method == 'kendall':
        measure_per_tree, _ = stats.kendalltau(list1, list2)
    elif method == 'weighted_kendall':
        measure_per_tree, _ = stats.weightedtau(list1, list2, rank=None)
    elif method == 'rbo':
        measure_per_tree = rbo_tau(list1, list2, p=1.0, weight_rank=False)
    elif method == 'weighted_rbo':
        measure_per_tree = rbo_tau(list1, list2, weight_rank=True)
    elif method == 'rbo_10':
        measure_per_tree = rbo_tau(list1, list2, sample_size=int(len(list2) * 0.1))
    elif method == 'rbo_5':
        measure_per_tree = rbo_tau(list1, list2, sample_size=int(len(list2) * 0.05))
    elif method == 'rbo_20':
        measure_per_tree = rbo_tau(list1, list2, sample_size=int(len(list2) * 0.2))
    elif method == 'weighted_rbo_5':
        measure_per_tree = rbo_tau(list1, list2, weight_rank=True, sample_size=int(len(list2) * 0.05))
    elif method == 'weighted_rbo_10':
        measure_per_tree = rbo_tau(list1, list2, weight_rank=True, sample_size=int(len(list2) * 0.1))
    elif method == 'Spearmanr':
        measure_per_tree, _ = stats.spearmanr(list1, list2)

    return measure_per_tree

def load_dataset_shuffled(fname):
    # import ipdb
    # ipdb.set_trace()
    data = pd.read_csv(fname, header=0)
    data = data.sample(frac=1)
    y = data['label']
    X = data.drop('label', axis=1)
    return X, y

def csv_to_svm():
    from sklearn.datasets import dump_svmlight_file
    import os

    fnames = os.listdir('./datasets/klf/')
    print(fnames)
    fnames = sorted(fnames)

    # read csv file
    for dataset in fnames:
        fname = os.path.join('./datasets/klf/', dataset)
        X, y = load_dataset_shuffled(fname)

        # convert csv to svm
        dump_svmlight_file(X, y, f="./datasets/svm_formated/"+dataset+".svm")
if __name__ == '__main__':
    csv_to_svm()

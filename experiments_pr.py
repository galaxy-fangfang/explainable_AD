import pickle as pkl
import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from rhf import RHF
import rbo
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

attributes = dict()

def traverse_rhf_tree(root, tree_scores, rhf_indices_top, rhf_ranking_top):
    if root.split_feature is None:
        return
    if root.split_feature.split('-')[0] not in attributes.keys():
        attributes[root.split_feature.split('-')[0]] = [root.depth]
    else:
        attributes[root.split_feature.split('-')[0]].append(root.depth)

    left_index = root.left.index
    left_size = len(left_index)
    p = left_size
    if left_size == 0:
        return 0.
    tree_scores[left_index] = p

    right_index = root.right.index
    right_size = len(root.right.index)
    p = right_size
    if right_size == 0:
        return 0.
    tree_scores[right_size] = p

    tree_indices = np.argsort(tree_scores)
    tree_ranking = np.argsort(tree_indices)
    tree_ranking_top = tree_ranking[rhf_indices_top]
    rhf_indices_top, rhf_ranking_top


    best_tree_rbo = rbo.RankingSimilarity(rhf_ranking_top, tree_ranking_top).rbo()
    print('best_tree_rbo: %.4f: ' % (root.ntree_ranking_top))

    if root.left is not None:
        traverse_rhf_tree(root.left)
    if root.right is not None:
        traverse_rhf_tree(root.right)

def prec_reca_k(scores, score_per_tree):
    scores_sorted = np.argsort(-scores)
    score_per_tree_sorted = np.argsort(-score_per_tree)
    prec = []
    reca = []

    for k in np.arange(1, len(scores), 10):
        y_true = scores.copy()
        y_pred = score_per_tree.copy()

        y_true[scores_sorted[:k]] = 1
        y_true[scores_sorted[k:]] = 0
        y_pred[score_per_tree_sorted[:k]] = 1
        y_pred[score_per_tree_sorted[k:]] = 0

        p = precision_score(y_true, y_pred, average='binary')
        r = recall_score(y_true, y_pred, average='binary')

        prec.append(p)
        reca.append(r)

    plt.plot(prec, reca, 'r--', label='p-r')
    plt.title('p-r')
    plt.savefig('./results/p_r/p_r.jpg')
    plt.cla()

    return prec, reca

def calap(prec, recall):
    mrec = [0] + recall + [1]
    mpre = [0] + prec + [0]

    for i in range(len(mpre) - 2, 0, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    ap = 0
    for i in range(len(mpre) - 1):
        if mpre[i + 1] > 0:
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]

    return ap

if __name__ == '__main__':
    # rhf model loaded
    datasets = ['penglobal', 'ionosphere', 'vowels_odds', 'mnist', 'musk']
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
        result = []
        print('~~~~~~~~~~~~~~~~~~~~~~dataset: ', dataset)
        rhf_path = './results/models/rhf_%s.pkl' % dataset
        if not os.path.exists(rhf_path):
            continue
        rhf = pkl.load(open(rhf_path, 'rb'))

        # dataset loaded
        dataset_path = os.path.join('./datasets/klf/', dataset)
        df = pd.read_csv(dataset_path, header=0)
        y = df["label"]
        sample_size = y.value_counts()[1]

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
        ap = average_precision_score(y, rhf_scores)

        # find the best tree in rhf
        best_ap = 0
        best_pr = 0

        for score_per_tree in score_per_tree_list:
            ap_per_tree = average_precision_score(y, score_per_tree)
            if ap_per_tree > best_ap:
                best_ap = ap_per_tree
            prec, reca = prec_reca_k(scores, score_per_tree)
            ap = calap(prec, reca)
            if ap > best_ap:
                best_pr = ap
                best_ap = ap_per_tree


        result.extend([df.shape[0], sample_size, ap, best_ap])

        results.append(result)
        datasets_saved.append(dataset)


    results = np.array(results)
    results = results.T
    data = pd.DataFrame(results, columns=datasets_saved, index=['n', 'anomaly', 'rhf_ap', 'ap_best_pr'])

    ap_all = ['rhf_ap', 'ap_best_pr']
    for ap_i in ap_all[1:]:
        data.loc[ap_i + '_gap'] = data.loc['rhf_ap'] - data.loc[ap_i]
    for ap_i in ap_all[1:]:
        data.loc[ap_i + '_gap/%'] = data.loc[ap_i + '_gap'] / data.loc['rhf_ap'] * 100
        data.round({ap_i + '_gap/%': 4})

    data['mean'] = data.mean(axis=1)
    data['var'] = data.var(axis=1)
    data.to_csv('./results/statfile/data_pr.csv')










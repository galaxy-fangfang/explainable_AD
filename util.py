import numpy as np
import rbo
import scipy.stats as stats
from sklearn.metrics import average_precision_score

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

def kendall_tau(rhf_scores, score_per_tree, weighted=False):
    if not weighted:
        tau, p_value = stats.kendalltau(rhf_scores, score_per_tree)
    else:
        tau, p_value = stats.weightedtau(rhf_scores, score_per_tree, rank=None)

    tau = (tau + 1) / 2.
    return tau

def rbo_tau(rhf_scores, score_per_tree, p=1.0, sample_size=None, weight_rank=False):
    # compute rbo for best tree
    rhf_indices = np.argsort(-rhf_scores)
    rhf_ranking = np.argsort(rhf_indices)

    per_tree_ranking = np.argsort(np.argsort(-score_per_tree))
    if sample_size is None:
        # import ipdb
        # ipdb.set_trace()
        per_tree_rbo = rbo.RankingSimilarity(np.arange(len(rhf_scores)), per_tree_ranking[rhf_indices]).rbo(p=p, weight_rank=weight_rank)
    else:
        rhf_indices_top = rhf_indices[:sample_size]
        rhf_ranking_top = rhf_ranking[rhf_indices_top]
        per_tree_ranking_top = per_tree_ranking[rhf_indices_top]
        per_tree_rbo = rbo.RankingSimilarity(rhf_ranking_top, per_tree_ranking_top).rbo(p=p, weight_rank=weight_rank)
    return per_tree_rbo

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
            elif method == 'rbo_5':
                measure_per_tree = rbo_tau(rhf_scores, score_per_tree, sample_size=int(len(score_per_tree) * 0.05))
                # measure_per_tree = rbo_tau(rhf_scores, score_per_tree, sample_size=sample_size)
            elif method == 'weighted_rbo_5':
                measure_per_tree = rbo_tau(rhf_scores, score_per_tree, weight_rank=True, sample_size=int(len(score_per_tree) * 0.05))
            elif method == 'weighted_rbo_10':
                measure_per_tree = rbo_tau(rhf_scores, score_per_tree, weight_rank=True, sample_size=int(len(score_per_tree) * 0.1))

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

def compute_measure(rhf_scores, score_per_tree_list, select_method=['weighted_kendall']):
    measures = {}
    for i, method in enumerate(select_method):
        measures[method] = np.zeros(len(score_per_tree_list))
        for tree_id, score_per_tree in enumerate(score_per_tree_list):
            if method == 'kendall':
                measure_per_tree, _ = stats.kendalltau(rhf_scores, score_per_tree)
            elif method == 'weighted_kendall':
                measure_per_tree, _ = stats.weightedtau(rhf_scores, score_per_tree, rank=None)
            elif method == 'rbo':
                measure_per_tree = rbo_tau(rhf_scores, score_per_tree, p=1.0, weight_rank=False)
            elif method == 'weighted_rbo':
                measure_per_tree = rbo_tau(rhf_scores, score_per_tree, weight_rank=True)
            elif method == 'rbo_10':
                measure_per_tree = rbo_tau(rhf_scores, score_per_tree, sample_size=int(len(score_per_tree) * 0.1))
                # measure_per_tree = rbo_tau(rhf_scores, score_per_tree, sample_size=sample_size * 2)
            elif method == 'rbo_5':
                measure_per_tree = rbo_tau(rhf_scores, score_per_tree, sample_size=int(len(score_per_tree) * 0.05))
                # measure_per_tree = rbo_tau(rhf_scores, score_per_tree, sample_size=sample_size)
            elif method == 'weighted_rbo_5':
                measure_per_tree = rbo_tau(rhf_scores, score_per_tree, weight_rank=True, sample_size=int(len(score_per_tree) * 0.05))
            elif method == 'weighted_rbo_10':
                measure_per_tree = rbo_tau(rhf_scores, score_per_tree, weight_rank=True, sample_size=int(len(score_per_tree) * 0.1))
            elif method == 'Spearmanr':
                measure_per_tree, _ = stats.spearmanr(rhf_scores, score_per_tree)

            # measures[i][tree_id] = measure_per_tree
            measures[method][tree_id] = measure_per_tree
    return measures

def spearmanr(rhf_scores, score_per_tree):
    return stats.spearmanr(rhf_scores, score_per_tree)
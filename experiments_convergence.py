import argparse
import pandas as pd
from sklearn.metrics import average_precision_score
import os
from statistics import mean
from time import time
from rhf import RHF
import pickle as pkl
from util import *
import matplotlib.pyplot as plt
import seaborn as sns

def run(datasets):
    #for dataset in os.listdir('./datasets/klf'):
    datasets = sorted(datasets)
    print(datasets)

    num_trees = 1000
    ap_datasets = []
    select_method = ['kendall', 'weighted_kendall', 'rbo', 'weighted_rbo', 'Spearmanr']
    color = {'kendall': 'green', 'weighted_kendall': 'red', 'rbo': 'skyblue', 'weighted_rbo': 'yellow',
             'Spearmanr': 'blue'}
    for dataset in datasets:
        print('dataset: ', dataset)
        dataset_path = os.path.join('./datasets/klf', dataset)
        df = pd.read_csv(dataset_path, header=0)#, nrows =100)
        y = df["label"]
        print('Dataset labels: ', y.value_counts())
        proportion = y.value_counts()[1] / (y.value_counts()[1] + y.value_counts()[0])
        # print('Y labels: ', y.values)
        X_raw = df.drop("label", axis=1)
        nominal_features = []
        numerical_features = []
    
        depth = 5
        X_raw.columns = [int(x) for x in X_raw.columns]
        for col in X_raw.columns:
            numerical_features.append(col)
    
        X = pd.get_dummies(data=X_raw, columns=nominal_features)
        nominal_dummy_features = {}
        for col in X.columns:
            if col not in numerical_features:
                unique_values = X[col].unique().tolist()
                nominal_dummy_features[col] = unique_values
        attrs = []
        for column in X.columns:
            attrs.append(column)
    
        # surrogate trees
        trials = [i for i in range(1)]
        ap_with_gt = []
        m_depth = []
    
        print("Dataset: ", dataset, 'proportion: ', proportion)
        time0 = time()
        # RHF
        black_box = RHF(num_trees=num_trees, max_height=5, seed_state=10007)
        black_box.fit(X)
        scores, scores_all = black_box.get_scores()
        rhf_scores = scores / num_trees
        # print(rhf_scores, scores_all)
        time_rhf = time()
        print('Time for rhf predict: ', time_rhf - time0)
        rhf_ap = average_precision_score(y, rhf_scores)
        print('Ap for rhf: ', rhf_ap)

        # find the best tree in rhf
        measures = compute_measure(rhf_scores, scores_all, select_method=select_method)

        # aps = np.zeros((len(select_method), num_trees))
        

        aps = {}
        # for i in range(len(select_method)):
        for method, measure in measures.items():
            # measure = measures[i]
            measure_sorted_index = np.argsort(-measure)
            score_sorted_current = np.zeros(len(scores_all[0]))
            aps[method] = np.zeros(len(scores_all))
            for j, index in enumerate(measure_sorted_index):
                score_sorted_current = (score_sorted_current * j + scores_all[index]) / (j + 1)
                average_precision_current = average_precision_score(y, score_sorted_current)
                gap_tmp = average_precision_current / rhf_ap * 100

                aps[method][j] = gap_tmp

        ap_datasets.append(aps)

    new_aps = {}
    import ipdb
    ipdb.set_trace()
    import json
    show_num_trees = 50
    x = range(0, show_num_trees)

    # json_handle = open('./results/json_files/ap_k.json', 'w')
    # json.dump(ap_datasets, json_handle)
    for i, method in enumerate(select_method):
        new_aps[method] = np.zeros(show_num_trees)
        for j in range(show_num_trees):
            new_aps[method][j] = 0
            for ap_dataset in ap_datasets:
                new_aps[method][j] += ap_dataset[method][j]
            new_aps[method][j] /= len(datasets)
        plt.plot(x, new_aps[method], color[method], label=method)
    
    plt.legend(loc='upper right')

    plt.savefig('./results/ap-trees/all_datasets.jpg')
    plt.cla()







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--datasets', required=True, nargs='+')
    # args = parser.parse_args()
    # datasets = args.datasets
    datasets = os.listdir('./datasets/klf')
    run(datasets)

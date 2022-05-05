import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_squared_error
from rhf import RHF
from util import *
import argparse
import time
from sklearn.model_selection import KFold, train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import IsolationForest
import seaborn as sns
from reg_tree.regressor import *
from easydict import EasyDict
from mpl_toolkits.mplot3d import Axes3D
import math


def run_regressor(config):
    datasets = config.datasets
    methods = config.methods
    num_iterations = config.num_iterations
    rhf_num_trees = config.rhf_num_trees
    rhf_max_height = config.rhf_max_height

    results = []
    for dataset in datasets:
        # dataset loaded
        dataset_path = os.path.join('./datasets/klf/', dataset)
        df = pd.read_csv(dataset_path, header=0)
        df = df.sample(frac=1).reset_index(drop=True)
        y = df["label"]
        sample_size = y.value_counts()[1]
        X = df.drop("label", axis=1)
        X.columns = [int(x) for x in X.columns]
        print('~~~~~~~~~~~~~~~~~~~~~~dataset: ', dataset)
        print('n=', len(y), ' d=', len(X.columns), ' anomaly=', sample_size)

        best_ap = {}
        for method in methods:
            best_ap[method] = []
        
        for d in range(num_iterations):
            result = []

            ### RHF in CYTHON VERSION
            black_box = RHF(num_trees=rhf_num_trees, max_height=rhf_max_height, seed_state=-1, check_duplicates=True, decremental=False,
                            use_kurtosis=True)
            black_box.fit(X)
            rhf_scores, scores_all = black_box.get_scores()
            rhf_scores = rhf_scores / rhf_num_trees
            rhf_scores = (rhf_scores - np.min(rhf_scores)) / (np.max(rhf_scores) - np.min(rhf_scores))
            rhf_ap = average_precision_score(y, rhf_scores)
            best_ap['rhf_ap'].append(rhf_ap)
            print('RHF AP for %s: %.4f ' % (dataset, rhf_ap))
            

            # EXPLAINER Cart Tree
            if 'reg_ap' in methods:
                explainer = DecisionTreeRegressor(max_depth=5)
                explainer.fit(X, rhf_scores)
                reg_scores = explainer.predict(X)
                reg_ap = average_precision_score(y, reg_scores)
                mse = mean_squared_error(rhf_scores, reg_scores)
                best_ap['reg_ap'].append(reg_ap)
                print('reg: %.4f' % reg_ap, ' mse: %f' % mse)

            # Save results: n, d, anomaly, list of best_ap dict keys, ID, dataset
            result.extend([df.shape[0], len(X.columns), sample_size])
            for method in best_ap.keys():
                result.append((best_ap[method][d]))
            result.extend([d, dataset.split('.')[0]])
            results.append(result)

    results = np.array(results)
    data_columns = ['n', 'd', 'anomaly'] + list(best_ap.keys()) + ['ID', 'Datasets']
    data = pd.DataFrame(results, columns=data_columns)
    csv_file = './results/rhf_regression_%s.csv' % datasets[0]
    data.to_csv(csv_file, index=False)
    if config.draw_boxplot == True:
        draw_plot(csv_file, config.methods)

def draw_plot(csv_file, methods):
    df_all = pd.read_csv(csv_file, header=0)

    print(df_all.columns)
    select_columns = methods + ['ID', 'Datasets']

    df =df_all[select_columns]

    # plot results for each dataset
    df1 = df.melt(id_vars=['Datasets', 'ID'])
    df1.columns = ['Datasets', 'ID', 'Group', 'AP']
    plt.figure(figsize=(15, 10))
    plt.title('Results on each dataset')
    sns.boxplot(data=df1, x='Datasets', y='AP', palette="Set2", hue='Group', width=0.8)
    plt.xticks(rotation=30)
    plt.savefig('./results/rhf_regression_%s.jpg' % datasets[0], dpi=300, bbox_inches='tight')
    plt.cla()
    

    # plot overall results for all datasets
    plt.title('Overall results on %d datasets' % int(df_all.shape[0]/10))
    df2 = df.groupby(['ID']).mean()
    sns.boxplot(data=df2, palette="Set2")
    plt.xticks(rotation=30)
    plt.savefig('./results/rhf_regression_%s_overall.jpg' % datasets[0], dpi=300, bbox_inches='tight')
    plt.cla()
    print(df2.mean(axis=0))

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', required=True, nargs='+')
    args = parser.parse_args()
    datasets = args.datasets
    datasets= sorted(datasets)
    print(datasets)

    config = EasyDict()
    config.datasets = datasets
    config.num_workers = 1
    config.num_iterations = 10

    config.rhf_num_trees = 100
    config.rhf_max_height = 5
    config.draw_boxplot = True
    config.methods = ['rhf_ap', 'reg_ap']
    run_regressor(config)










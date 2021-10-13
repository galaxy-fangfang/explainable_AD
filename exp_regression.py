import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import os
from statistics import mean
from time import time
from rhf import RHF
import pickle as pkl
import pandas as pd
from reg_tree.staci import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m

def bound_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

def run(datasets):
    datasets = sorted(datasets)
    print(datasets)
    results = []
    datasets_saved = []
    ap_train_datasets = []
    ap_test_datasets = []
    rhf_aps_datasets = []

    for dataset in datasets:
        result = []
        print('dataset: ', dataset)
        dataset_path = os.path.join('./datasets/klf', dataset)
        df = pd.read_csv(dataset_path, header=0)#, nrows =100)
        y = df["label"]
        sample_size = y.value_counts()[1]

        X_raw = df.drop("label", axis=1)
        nominal_features = []
        numerical_features = []

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
    

        # RHF
        count = 10
        rhf_aps = []
        ap_trains = []
        ap_train_gap = []
        ap_tests = []
        # ap_test_gap = np.zeros(count)
        nodes_rhf_unique = []
        nodes_reg = []
        nodes_reg_unique = []

        # for i in range(count):
        #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        kf = KFold(n_splits=count, shuffle=True)
        for train, test in kf.split(X):
            X_train, X_test, y_train, y_test = X.loc[train], X.loc[test], y.loc[train], y.loc[test]

            while np.sum(y_train) == 0 or np.sum(y_test) == 0:
                # continue
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

            #RHF
            black_box = RHF(num_trees=100, max_height=5, seed_state=-1, check_duplicates=True, decremental=False,
                            use_kurtosis=True)
            black_box.fit(X_train)
            y_pred, _ = black_box.get_scores()
            y_pred = y_pred / 100
            nodes_rhf_unique.append(len(np.unique(y_pred)))
            # NORMALIZE
            y_pred = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred))
            rhf_ap = average_precision_score(y_train, y_pred)
            rhf_aps.append(rhf_ap)

            # explainer
            explainer = DecisionTreeRegressor(max_depth=5)
            explainer.fit(X_train, y_pred)
            exp_predict_train = explainer.predict(X_train)
            ap_train = average_precision_score(y_train, exp_predict_train)
            ap_trains.append(ap_train)
            ap_train_gap.append((rhf_ap - ap_train) / rhf_ap * 100)

            exp_predict_test = explainer.predict(X_test)
            ap_tests.append(average_precision_score(y_test, exp_predict_test))

            nodes_reg.append(explainer.get_n_leaves())
            nodes_reg_unique.append(len(np.unique(exp_predict_test)))

            # ap_test_gap[i] = (rhf_aps[i] - ap_test[i]) / rhf_aps[i] * 100

        rhf_aps_mean, rhf_aps_std = mean_confidence_interval(rhf_aps)
        ap_trains_mean, ap_trains_std = mean_confidence_interval(ap_trains)
        ap_tests_mean, ap_tests_std = mean_confidence_interval(ap_tests)
        rhf_aps_datasets.append(rhf_aps)
        ap_train_datasets.append(ap_trains)
        ap_test_datasets.append(ap_tests)

        print('AP rhf: ', np.mean(rhf_aps), rhf_aps_mean)
        print("Surrogate Ap for train data: ",  ap_trains_mean, ap_trains_std)
        # print('gap: ', np.mean(ap_train_gap))
        print('Surrogate Ap for test data: ', ap_tests_mean, ap_tests_std)
        print('Average nodes for RHF: ', np.mean(nodes_rhf_unique))
        print('Average nodes for regression tree: ', np.mean(nodes_reg))

        result.extend([df.shape[0], sample_size, rhf_aps_mean, rhf_aps_std, ap_trains_mean, ap_trains_std, ap_tests_mean, ap_tests_std, np.mean(nodes_rhf_unique), np.mean(nodes_reg)])
        results.append(result)
        datasets_saved.append(dataset)
    rhf_aps_10runs = np.mean(rhf_aps_datasets, axis=0)
    rhf_aps_10runs_mean, rhf_aps_10runs_std = mean_confidence_interval(rhf_aps_10runs)
    ap_train_10runs = np.mean(ap_train_datasets, axis=0)
    ap_train_10runs_mean, ap_train_10runs_std = mean_confidence_interval(ap_train_10runs)
    ap_test_10runs = np.mean(ap_test_datasets, axis=0)
    ap_test_10runs_mean, ap_test_10runs_std = mean_confidence_interval(ap_test_10runs)

    print('all datasets rhf: ', rhf_aps_10runs_mean, rhf_aps_10runs_std)
    print('all datasets train: ', ap_train_10runs_mean, ap_train_10runs_std)
    print('all datasets test: ', ap_test_10runs_mean, ap_test_10runs_std)

    results = np.array(results)
    results = results.T
    data_index = ['n', 'anomaly', 'rhf_ap_mean', 'rhf_ap_std', 'surrogate_train_ap_mean', 'surrogate_train_ap_std', 'surrogate_test_ap_mean', 'surrogate_test_ap_std', 'leafs for rhf', 'leafs for surrogate']
    data = pd.DataFrame(results, columns=datasets_saved, index=data_index)
    data.to_csv('./results/statfile/data_regress.csv', float_format='%.4f', index=False)

    ap_train_datasets = np.array(ap_train_datasets).T
    data = pd.DataFrame(ap_train_datasets, columns=datasets_saved)
    data.to_csv('./results/statfile/data_regress_surrogate_train_ap.csv', float_format='%.4f', index=False)

    ap_test_datasets = np.array(ap_test_datasets).T
    data = pd.DataFrame(ap_test_datasets, columns=datasets_saved)
    data.to_csv('./results/statfile/data_regress_surrogate_test_ap.csv', float_format='%.4f', index=False)

    rhf_aps_datasets = np.array(rhf_aps_datasets).T
    data = pd.DataFrame(rhf_aps_datasets, columns=datasets_saved)
    data.to_csv('./results/statfile/data_regress_rhf_aps.csv', float_format='%.4f', index=False)

def draw_plots():
    df_rhf = pd.read_csv('./results/statfile/data_regress_rhf_aps.csv', header=0)
    df_test = pd.read_csv('./results/statfile/data_regress_surrogate_test_ap.csv', header=0)
    df_train = pd.read_csv('./results/statfile/data_regress_surrogate_train_ap.csv', header=0)

    # 直接用dataframe的自带函数画图，但是boxplot画的是数据的分布图，不是置信区间
    df_mean = df_rhf.mean(axis=0)
    df_mean = df_mean.sort_values(ascending=False)
    dataset_sorted = list(df_mean.index)
    plt.figure(figsize=(25, 11))
    # plt.xlabel('datasets')

    plt.title('Ap of rhf on trainset')
    plt.ylabel('average precision')
    df_rhf.boxplot(column=dataset_sorted, rot=45, fontsize=11)
    plt.savefig('results/regression_method/rhf_ap_boxplot.jpg')
    plt.cla()

    plt.title('Ap of regression tree on trainset')
    plt.ylabel('average precision')
    df_train.boxplot(column=dataset_sorted, rot=45, fontsize=11)
    plt.savefig('results/regression_method/reg_train_boxplot.jpg')
    plt.cla()

    plt.title('Ap of regression tree on testset')
    plt.ylabel('average precision')

    df_test.boxplot(column=dataset_sorted, rot=45, fontsize=11)
    plt.savefig('results/regression_method/reg_test_boxplot.jpg')
    plt.clf()


    # 用plot画3个的置信区间，error bar
    # df_rhf_mean = df_rhf.mean(axis=1)
    # df_train_mean = df_train.mean(axis=1)
    # df_test_mean = df_test.mean(axis=1)
    # df_all = pd.DataFrame(df_rhf_mean, columns=['rhf on trainset'])
    # df_all['reg on trainset'] = df_train_mean
    # df_all['ref on testset'] = df_test_mean
    #
    # mean = df_all.agg(mean_confidence_interval)
    # bound = df_all.agg(bound_confidence_interval)
    # # matplotlib.rcParams.update({'font.size': 24})
    # # plt.figure(figsize=(20, 7))
    # # plt.xticks(rotation=90)
    # plt.title('confidence interval with 95%')
    # plt.ylabel('average precision')
    # plt.errorbar(df_all.columns, mean, yerr=bound, linestyle='None', marker='o', markersize=4, capsize=3)
    # plt.savefig('results/regression_method/errorbar_ci_plot.jpg')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--datasets', required=True, nargs='+')
    # args = parser.parse_args()
    # datasets = args.datasets
    datasets = os.listdir('./datasets/klf')
    # datasets = ['abalone']
    # run(datasets)
    draw_plots()

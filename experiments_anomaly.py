import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import os
from statistics import mean
from time import time
from rhf import RHF
import pickle as pkl
# from interpret import show
# from interpret.data import ClassHistogram
# from interpret.glassbox import ExplainableBoostingClassifier, LogisticRegression, ClassificationTree, DecisionListClassifier



# datasets = ['http_logged']
# datasets = ['abalone']
# datasets = ['penglobal',  'ionosphere', 'vowels_odds','mnist', 'musk']
# datasets = ['ionosphere', 'vowels_odds', 'http_logged']
# datasets = {'penglobal': 0.111,  'ionosphere': 0.358, 'vowels_odds': 0.034, 'mnist': 0.092, 'musk': 0.031}
# datasets = ['penglobal',  'ionosphere', 'vowels_odds', 'mnist', 'musk']
# datasets = {'mnist': 0.1}
# datasets = {'penglobal': 0.2}
# datasets = {'vowels_odds': 0.05}
# datasets = {'musk': 0.05}

# for dataset in datasets:
def run(datasets):
    #for dataset in os.listdir('./datasets/klf'):
    print(datasets)
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
        split_criterion = 'Kendall'#'RBO'
        print('split-criterion: ', split_criterion)
        time0 = time()
        # RHF
        count = 10
        for i in range(count):
            black_box = RHF(num_trees=1000, max_height=5, split_criterion='kurtosis', dataset=dataset)
            y_pred, rhf_trees = black_box.fit(X, y)
            # time_rhf = time()
            # print('Time for rhf predict: ', time_rhf - time0)
            average_precision = average_precision_score(y, y_pred)
            print('Ap for rhf: ', average_precision)
            fhandle = open('./results/models/1000_trees/rhf_%s_%d.pkl' % (dataset, i), 'wb')
            pkl.dump(rhf_trees, fhandle)
            fhandle.close()
    
        y_pred_df = pd.Series((v for v in y_pred), name="target", index=X.index)
        # for trial in trials:
        #
        #     # microsoft explainer
        #     # ebm = ExplainableBoostingClassifier(random_state=1, n_jobs=-1)
        #     # ebm.fit(X, y_pred_df)  # Works on dataframes and numpy arrays
        #     # ebm_global = ebm.explain_global(name='EBM')
        #     # show(ebm_global)
        #
        #     # STACI explainer
        #     explainer = STACISurrogatesKendalTauDis(max_depth=depth, prune=False, split_criterion=split_criterion, proportion=proportion)#'R-squared')#'RBO')
        #     explainer.fit(X=X,
        #                   y=y_pred_df,
        #                   bb_model=black_box,
        #                   features=attrs,
        #                   target='target')
        #     time_surrogate = time()
        #     # print('Time for surrogate tree fit: ', time_surrogate - time_rhf)
        #
        #     fhandle = open('./results/models/surrogate_%s.pkl' % dataset, 'wb')
        #     pkl.dump(explainer.trees, fhandle)
        #     fhandle.close()
        #     # print('Forest has been saved in: ', './results/{}.pkl'.format('kendall_tau_surrogate_tree'))
        #
        #     # read tree from pkl file
        #     # explainer.trees = pkl.load(open('./results/tree.pkl', 'rb'))
        #
        #     # max_nodes = 0
        #     # max_leaf_nodes = 0
        #     # maximum_depth = 0
        #     # for dtree in explainer.trees:
        #     #     if len(dtree.nodes) > max_nodes:
        #     #         max_nodes = len(dtree.nodes)
        #     #     tree_depth = maxi_depth(dtree.nodes[0])
        #     #     if tree_depth > maximum_depth:
        #     #         maximum_depth = tree_depth
        #     #     for node in dtree.nodes:
        #     #         if isinstance(node, LeafNode):
        #     #             max_leaf_nodes += 1
        #     # time_max_depth = time()
        #     # print('Time for max depth: ', time_max_depth - time_surrogate)
        #
        #     # for t, value in enumerate(explainer.trees):
        #     #     print("Tree: ", t)
        #     #     for node in value.nodes:
        #     #         if isinstance(node, InternalNode):
        #     #             print("Node id: {}, Node feature: {}, Threshold: {}, Sizes: {}, Level: {}"
        #     #                   .format(node.node_id, node.feature, node.threshold, node.n_samples, node.depth))
        #     #         else:
        #     #             print("Leaf Node id: {}, Sizes: {}, Level: {}".format(node.node_id, node.n_samples, node.depth))
        #
        #     exp_predict = explainer.predict_RHF(X)[0]
        #     ap = average_precision_score(y, exp_predict)
        #
        #     # m_depth.append(maximum_depth)
        #     ap_with_gt.append(ap)
        #
        # # print("Maximal length: ", mean(m_depth))
        # # print('Maximal leaf nodes: ', max_leaf_nodes)
        # print("Ap for surrogate: ", mean(ap_with_gt))
        # print('Rhf prediction: ', np.unique(y_pred))
        # print('Surrogate prediction: ', np.unique(exp_predict))
        # print('Rhf prediction: ', y_pred)
        # print('Surrogate prediction: ', exp_predict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', required=True, nargs='+')
    args = parser.parse_args()
    datasets = args.datasets
    run(datasets)

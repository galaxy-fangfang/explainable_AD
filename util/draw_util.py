import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score
import os
def find_threshold(labels, output_scores):
    acc = []
    thresh = []
    start = min(output_scores) + 0.5
    end = max(output_scores) - 0.5

    for score in np.arange(start, end, 0.5):
        y_pred = output_scores.copy()
        y_pred[y_pred <= score] = 0
        y_pred[y_pred > score] = 1
        accuracy = accuracy_score(labels, y_pred)
        acc.append(accuracy)
        thresh.append(score)

    plt.plot(thresh, acc, 'r--', label='acc')
    plt.title('acc-threshold')
    plt.savefig('./results/acc.jpg')
    plt.cla()

    idx = np.argmax(acc)
    return acc[idx], thresh[idx]


def draw_histogram_anomaly_scores(labels, output_scores_ori, save_path):

    output_scores = output_scores_ori.copy()
    accuracy_thresh, thresh = find_threshold(labels, output_scores)
    # print('accuracy for random histogram forest: ', accuracy_thresh)

    # 画直方图
    plt.cla()

    output_score_label0 = []
    output_score_label1 = []

    for i, label in enumerate(labels.values):
        if label == 1:
            output_score_label1.append(output_scores[i])
        elif label == 0:
            output_score_label0.append(output_scores[i])
        else:
            pass

    # Sheng's plot
    # find the interval of scores
    scoreMin0, scoreMax0 = min(output_score_label0), max(output_score_label0)
    scoreMin1, scoreMax1 = min(output_score_label1), max(output_score_label1)
    # print(scoreMin0, scoreMax0, scoreMin1, scoreMax1)
    nslice = 50
    bins0 = np.linspace(scoreMin0, scoreMax0, nslice)
    bins1 = np.linspace(scoreMin1, scoreMax1, nslice)

    hist0, bin_edges0 = np.histogram(output_score_label0, bins=bins0)
    bin_edges0 = bin_edges0 + (bin_edges0[1] - bin_edges0[0]) / 2
    bin_edges0 = bin_edges0[:-1]
    hist1, bin_edges1 = np.histogram(output_score_label1, bins=bins1)
    bin_edges1 = bin_edges1 + (bin_edges1[1] - bin_edges1[0]) / 2
    bin_edges1 = bin_edges1[:-1]

    # plt.figure()
    plt.bar(bin_edges0, hist0, width=(bin_edges0[1] - bin_edges0[0]), color='red', edgecolor='black')
    plt.bar(bin_edges1, hist1, width=(bin_edges1[1] - bin_edges1[0]), color='blue', edgecolor='black')
    # plt.ylim([0, max()])


    plt.text(x=bin_edges1[np.argmax(hist1)],
             y=max(hist1),
             s='count: {:.0f}'.format(np.sum([np.array(output_scores) > thresh])),
             color='purple')
    plt.text(x=bin_edges0[np.argmax(hist0)],
             y=max(hist0),
             s='count: {:.0f}'.format(np.sum([np.array(output_scores) <= thresh])),
             color='purple')
    # plt.text(x=bin_edges1[-1],
    #          y=max(max(hist0), max(hist1)),
    #          s='threshold: {:.1f}, acc: {:.4f}'.format(thresh, accuracy_thresh),
    #          color='black')
    average_precision = average_precision_score(labels, output_scores)
    plt.text(x=max(bin_edges1[-1], bin_edges0[-1]),
             y=max(max(hist0), max(hist1)),
             s='ap: {:.4f}'.format(average_precision),
             color='black')

    plt.title('Distribution of anomaly scores')

    plt.savefig(save_path)
    plt.cla()

    output_scores[output_scores <= thresh] = 0
    output_scores[output_scores > thresh] = 1
    return output_scores, average_precision


if __name__ == '__main__':
    file_list = os.listdir('./results/draw_file/')
    datasets = ['abalone', 'aloi', 'annthyroid', 'arrhytmia', 'breastcancer', 'cardio', 'cover', 'http_all', 'http_logged', 'ionosphere', 'kdd99', 'kdd_finger', 'kdd_ftp', 'kdd_ftp_distinct', 'kdd_http', 'kdd_http_distinct', 'kdd_other', 'kdd_smtp', 'kdd_smtp_distinct', 'magicgamma', 'mammography', 'mnist', 'mulcross', 'musk', 'penglobal', 'pima_odds', 'satellite', 'satimages', 'shuttle_odds', 'smtp_all', 'spambase', 'thyroid', 'vertebral', 'vowels_odds', 'wbc', 'wikiqoe', 'wine', 'yeast']
    rhf_ap = [0.349861332, 0.036421793, 0.312303654, 0.426291297, 0.950886161, 0.583244578, 0.073345178, 0.755510374, 0.983653879, 0.812284897, 0.758900163, 0.270750312, 0.915089282, 0.406582485, 0.55914781, 0.770077212, 0.53646838, 0.402193695, 0.079218818, 0.632027151, 0.142778595, 0.356077706, 0.732346253, 0.992640481, 0.552217867, 0.490125852, 0.646576002, 0.927389901, 0.93251865, 0.94033359, 0.402493921, 0.522578938, 0.093924758, 0.129351148, 0.577839659, 0.115835915, 0.065248148, 0.233377534]
    apss = {}
    for i, dataset in enumerate(datasets):
        apss[dataset] = rhf_ap[i]
    for file_name in file_list:
        file_path = os.path.join('./results/draw_file/', file_name)
        file_data = pd.read_csv(file_path, header=0)
        dataset = file_name.split('.')[0].replace('distance_', '')
        distancess = file_data['distancess']
        num_clusters = file_data['num_clusters']
        ap_clusters = file_data['ap_clusters']

        fig, ax1 = plt.subplots()
        color = 'tab:red'

        ax1.set_xlabel('distances')
        ax1.set_ylabel('num_clusters', color=color)
        ax1.plot(distancess, num_clusters, 'o', color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('ap_clusters', color=color)  # we already handled the x-label with ax1
        ax2.plot(distancess, ap_clusters, 'o', color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        # plt.legend(loc='upper right')
        plt.title('%s_%.4f' % (dataset, apss[dataset]))

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig('./results/linka/fix_distance/%s.jpg' % dataset)
        plt.cla()
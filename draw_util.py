import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, average_precision_score

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

import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


"""
Visulization weights
"""

class Hooks():

    def __init__(self):
        self.input = []
        self.output = []
        self.classifier = []

    def hook(self, module, input, output):

        self.input = input[0].cpu().detach().numpy()
        self.output = output[0].cpu().detach().numpy()

    def hook_cls(self, module, input, output):

        self.input.append(input[0].cpu().detach().numpy())
        self.output.append(output.cpu().detach().numpy())

    def get_data(self):

        return self.input, self.output

    def get_classifier(self):
        return self.classifier


def visualize_data_distribution(data, names, y_true = None):
    dec = PCA(n_components=2)
    data2 = dec.fit_transform(data) # 2维数据
    print(y_true)
    plt.figure(dpi=300)
    for i in range(15):
        plt.scatter(data2[i, 1], data2[i, 0],  label=names[i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=60)


    plt.legend(loc='lower right', borderpad=0, handletextpad=0)
    plt.show()

def visualize_bounds(data, y_true, y_pred, title, label=False):

    # dec = TSNE(n_components=2, n_iter = 1000, verbose=1)
    dec = PCA(n_components=2, )
    data2 = dec.fit_transform(data) # 2维数据

    y_pred_cls = np.zeros_like(y_pred, dtype=np.int)
    y_pred_cls[y_pred >= 0.5] = 1  # 预测类别

    conf = []
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred_cls[i] == 1:
            conf.append(3)  # TP
        elif y_true[i] == 1 and y_pred_cls[i] == 0:
            conf.append(2)  # FN
        elif y_true[i] == 0 and y_pred_cls[i] == 1:
            conf.append(1)  # FP
        elif y_true[i] == 0 and y_pred_cls[i] == 0:
            conf.append(0)  # TN

    conf = np.array(conf)
    fig = plt.figure(dpi=300)
    ax = fig.subplots()

    if label:
        target = ['Negative sample', 'Positive sample']
        for i, color in zip([0, 1], ['lightseagreen', 'orangered']):
            idx = np.where(y_true == i)
            plt.scatter(data2[idx, 0], data2[idx, 1], c=color, label=target[i],
                        cmap=plt.cm.RdYlBu, edgecolor='black', s=15)
    else:

        target = ['TN sample', 'FP sample', 'FN sample', 'TP sample']
        for i, color in zip([0, 1, 2, 3], ['lightseagreen', 'greenyellow', 'gold', 'orangered']):
            idx = np.where(conf == i)
            plt.scatter(data2[idx, 0], data2[idx, 1], c=color, label=target[i],
                        cmap=plt.cm.RdYlBu, edgecolor='black', s=20)


    plt.legend(loc='upper right', borderpad=0.3, handletextpad=0, prop={'family': 'Times New Roman', 'size': 12})

    plt.title(title, fontdict={'family': 'Times New Roman', 'size': 13})

    plt.tick_params(labelsize=10)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.show()


def visualize_attention(attention, xlabel = None, ylabel = None, receive = True, save=""):

    print(attention.shape)
    fig = plt.figure(dpi=300)
    ax = fig.subplots()
    cmap = "Reds"

    if xlabel is not None and ylabel is not None:
        sns.heatmap(attention, xticklabels = xlabel, yticklabels = ylabel, cmap=cmap)

    elif ylabel is not None:
        sns.heatmap(attention, yticklabels = ylabel, cmap=cmap)

    else:
        sns.heatmap(attention)
    # use matplotlib.colorbar.Colorbar object
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=8)

    plt.tick_params(labelsize=10)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    if save =="":
        plt.show()
    else:
        plt.savefig(save)

    if receive:
        att = attention.sum(axis=0).reshape(-1, 1)
        df = pd.DataFrame(att, index=xlabel, columns=['w'])
        df.to_csv(f'results/{save[:-4]}_att_re.csv')

def visualize_attention_avg(attentions, xlabel = None, ylabel = None, receive = True, save=""):
    att = np.mean(attentions, axis=0)
    visualize_attention(att, xlabel, ylabel, receive, save)
    if receive:
        y = np.mean(att, axis=0)
        plt.figure(dpi=300)
        plt.bar(ylabel, y, edgecolor='black', color='blue')
        plt.xticks(size=8)
        if save != "" and receive:
            plt.savefig(f"{save[:-4]}_rc.png")


def visualize_func_residue_attention(attentions, funcs, seqs, save=""):
    """
    attentions: (C, R)
    map(func): residues
    """
    residues = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K',
            'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    r2idx = {}
    for i, r in enumerate(residues):
        r2idx[r] = i

    res = np.zeros((15, 20), dtype=np.float32)

    for i, att in enumerate(attentions):
        seq = seqs[i]
        if(len(seq) > 50):
            seq = seq[:25] + seq[-25:]
            assert(len(seq) == 50)
        for j, c in enumerate(att):
            # j: func idx
            # c: [R]
            for k in range(len(seq)):
                ridx = r2idx[seq[k]]
                value = c[k]
                res[j, ridx] += value
    res = res / len(attentions)

    visualize_attention(res, xlabel=list(residues), ylabel=funcs, receive=False, save=save)

def cos_smi_avg(data, label):

    smis = []
    for x in data:
        z = pd.DataFrame(x.transpose(), columns=label)
        smi = z.corr()
        smis.append(smi)
    smis = np.mean(np.array(smis), axis=0)

    plt.figure(dpi=300)
    # cmap = "Greys"
    cmap = "Blues"
    sns.heatmap(smis, yticklabels = label, xticklabels = label,cmap=cmap)
    plt.show()

def cos_smi(data, label):
    c, h = data.shape

    smi = cosine_similarity(data)
    # for i in range(c):
    #     smi[i][i] = 0

    print(smi)
    fig = plt.figure(dpi=300)
    ax = fig.subplots()
    cmap = "Reds"

    sns.heatmap(smi, yticklabels = label, xticklabels = label,cmap=cmap)
    plt.show()



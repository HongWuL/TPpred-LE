from utils.util_methods import *
from utils.load_data import load_seqs_and_labels
import numpy as np
import os
import pandas as pd
import shutil

pts = ['AMP.txt', 'TXP.txt', 'ABP.txt', 'AIP.txt', 'AVP.txt',
       'ACP.txt', 'AFP.txt', 'DDV.txt', 'CPP.txt', 'CCC.txt',
        'APP.txt', 'AAP.txt', 'AHTP.txt', 'PBP.txt', 'QSP.txt']

names = [pt[:-4] for pt in pts]

train_seqs, train_labels = load_seqs_and_labels('../out90v4/train', names)

print(train_labels)
N = np.sum(train_labels)
print(N)

n, m = train_labels.shape

pos_idx = []
for i in range(n):
    for j in range(m):
        if train_labels[i][j] == 1:
            pos_idx.append(m * i + j)


for p in range(1, 6):
    # 将p的正样本标签随机置为0
    np.random.seed(123)
    z_idx = np.random.choice(pos_idx, size=int(N * p * 0.1), replace=False)
    p_labels = np.copy(train_labels)
    for z in z_idx:
        i = z // 15
        j = z % 15
        assert p_labels[i][j] == 1
        p_labels[i][j] = 0

    print(np.sum(p_labels))
    pth = f'p{p}/train/'
    if not os.path.exists(pth):
        os.makedirs(pth)

    df_label_trian = pd.DataFrame(data=p_labels, columns=names)
    df_label_trian.to_csv(os.path.join(pth, 'labels.csv'), index=False)
    shutil.copyfile('../out90v4/train/seqs.fasta', os.path.join(pth, 'seqs.fasta'))



val_seqs, val_labels = load_seqs_and_labels('../out90v4/val', names)

print(val_labels)
N = np.sum(val_labels)
print(N)

n, m = val_labels.shape

pos_idx = []
for i in range(n):
    for j in range(m):
        if val_labels[i][j] == 1:
            pos_idx.append(m * i + j)


for p in range(1, 6):
    # 将p的正样本标签随机置为0
    np.random.seed(123)
    z_idx = np.random.choice(pos_idx, size=int(N * p * 0.1), replace=False)
    p_labels = np.copy(val_labels)
    for z in z_idx:
        i = z // 15
        j = z % 15
        assert p_labels[i][j] == 1
        p_labels[i][j] = 0

    print(np.sum(p_labels))
    pth = f'p{p}/val/'
    if not os.path.exists(pth):
        os.makedirs(pth)

    df_label_val = pd.DataFrame(data=p_labels, columns=names)
    df_label_val.to_csv(os.path.join(pth, 'labels.csv'), index=False)
    shutil.copyfile('../out90v4/val/seqs.fasta', os.path.join(pth, 'seqs.fasta'))
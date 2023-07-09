import numpy as np
from sklearn.metrics import roc_auc_score, auc, accuracy_score, precision_recall_curve, f1_score, balanced_accuracy_score, \
    recall_score, precision_score, matthews_corrcoef, multilabel_confusion_matrix
from sklearn.metrics import hamming_loss
import os
import pandas as pd

def instances_overall_metrics(y_pred: np.array, y_true: np.array, threshold=0.5, save = None, show = True):
    """
    计算样本层面的整体评价指标
    """
    y_pred_cls = np.zeros_like(y_pred, dtype=np.int)
    y_pred_cls[y_pred > threshold] = 1    # 预测类别

    n, m = y_true.shape

    # Hamming Loss
    HLoss = hamming_loss(y_true, y_pred_cls)

    # Accuracy
    ACC = 0
    for i in range(n):
        ACC += (np.sum((y_pred_cls[i] == 1) & (y_true[i] == 1)) / np.sum((y_pred_cls[i] == 1) | (y_true[i] == 1)))
    ACC /= n

    # Precision
    Precision = 0
    for i in range(n):
        if (np.sum((y_pred_cls[i] == 1) & (y_true[i] == 1)) == 0): continue
        Precision += (np.sum((y_pred_cls[i] == 1) & (y_true[i] == 1)) / np.sum(y_pred_cls[i] == 1) )
    Precision /= n

    # Recall
    Recall = 0
    for i in range(n):
        Recall += (np.sum((y_pred_cls[i] == 1) & (y_true[i] == 1)) / np.sum(y_true[i] == 1))
    Recall /= n

    # Absolute ture
    AT = 0
    for i in range(n):
        if(np.all(y_pred_cls[i] == y_true[i])):
            AT += 1
    AT /= n

    df = pd.DataFrame({'HLoss': [HLoss], 'Accuracy': [ACC], 'Precision': [Precision], 'Recall': [Recall], 'Absolute true': [AT]})
    if show:
        print(df)

    if save is not None:
        df.to_csv(save)

    return df

def label_overall_metrics(y_pred: np.array, y_true: np.array, threshold=0.5, save = None, show = True):
    """
    计算macro和micro指标
    """
    n_samples, n_class = y_pred.shape
    pos = 1
    neg = 0

    y_pred_cls = np.zeros_like(y_pred)
    y_pred_cls[y_pred >= threshold] = 1    # 预测类别

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    res_acc = []
    res_auc = []
    res_mcc = []
    res_aupr = []
    res_precision = []
    res_recall = []
    res_f1 = []
    res_bacc = []
    res_rkcc = []
    

    for c in range(n_class):
        y_c = y_pred_cls[:, c]
        y_t = y_true[:, c]
        y_p = y_pred[:, c]

        tp = np.sum(np.logical_and(y_t == pos, y_c == pos))
        tn = np.sum(np.logical_and(y_t == neg, y_c == neg))
        fp = np.sum(np.logical_and(y_t == neg, y_c == pos))
        fn = np.sum(np.logical_and(y_t == pos, y_c == neg))

        TP += tp
        TN += tn
        FP += fp
        FN += fn

        F1 = f1_score(y_t, y_c)
        ACC = accuracy_score(y_t, y_c)
        AUC = roc_auc_score(y_t, y_p)
        precision, recall, thresholds = precision_recall_curve(y_t, y_p)
        AUPR = auc(recall, precision)
        BACC = balanced_accuracy_score(y_t, y_c)
        MCC = matthews_corrcoef(y_t, y_c)
        Recall = recall_score(y_t, y_c)
        Precision = precision_score(y_t, y_c)
        
        res_mcc.append(round(MCC, 3))
        res_acc.append(round(ACC, 3))
        res_auc.append(round(AUC, 3))
        res_aupr.append(round(AUPR, 3))
        res_recall.append(round(Recall, 3))
        res_precision.append(round(Precision, 3))
        res_f1.append(round(F1, 3))
        res_bacc.append(round(BACC, 3))

    ACC_micro = (TP + TN) / (TP + TN + FP + FN)
    MCC_micro = (TP * TN - FP * FN) / (np.sqrt(TP + FN) * np.sqrt(TP + FP) * np.sqrt(TN + FP) * np.sqrt(TN + FN))
    Precision_micro = TP / (TP + FP)
    Recall_micro = TP / (TP + FN)
    BACC_micro = ((TP / (TP + FN)) + (TN / (TN + FP))) / 2
    AUC_micro = roc_auc_score(y_true, y_pred, average='micro', multi_class='ovr')
    AUPR_micro = 0
    F1_micro = f1_score(y_true, y_pred_cls, average='micro')

    ACC_macro = np.mean(res_acc)
    MCC_macro = np.mean(res_mcc)
    Precision_macro = np.mean(res_precision)
    Recall_macro = np.mean(res_recall)
    BACC_macro = np.mean(res_bacc)
    AUC_macro = np.mean(res_auc)
    AUPR_macro = np.mean(res_aupr)
    F1_macro = np.mean(res_f1)


    df = pd.DataFrame({'ACC': [ACC_macro, ACC_micro], 'BACC': [BACC_macro, BACC_micro],
                       'AUC': [AUC_macro, AUC_micro], 'MCC': [MCC_macro, MCC_micro],
                       'AUPR': [AUPR_macro, AUPR_micro], 'F1': [F1_macro, F1_micro],
                       'Precision': [Precision_macro, Precision_micro], 'Recall': [Recall_macro, Recall_micro]},
                      index= ['macro','micro'])
    if show:
        print(df)
    if save is not None:
        df.to_csv(save)

    return df


def binary_metrics(y_pred: np.array, y_true: np.array, class_names, threshold=0.5, save = None, show = True):
    """
    计算每一类的准确率，精度, 召回率, MCC
    Args:
        y_pred: 预测得分, [n_samlpes, n_class]
        y_true: 真实类别, [n_samlpes, n_class]
    """
    n_samples, n_class = y_pred.shape
    pos = 1
    neg = 0

    y_pred_cls = np.zeros_like(y_pred)
    y_pred_cls[y_pred >= threshold] = 1    # 预测类别


    res_acc = []
    res_auc = []
    res_mcc = []
    res_aupr = []
    res_precision = []
    res_recall = []
    res_f1 = []
    res_bacc = []
    res_rkcc = []

    # multi-label confusion matrics (n_class)
    mcm = multilabel_confusion_matrix(y_true, y_pred_cls)

    for c in range(n_class):
        y_c = y_pred_cls[:, c]
        y_t = y_true[:, c]
        y_p = y_pred[:, c]

        tp = np.sum(np.logical_and(y_t == pos, y_c == pos))
        tn = np.sum(np.logical_and(y_t == neg, y_c == neg))
        fp = np.sum(np.logical_and(y_t == neg, y_c == pos))
        fn = np.sum(np.logical_and(y_t == pos, y_c == neg))

        F1 = f1_score(y_t, y_c)
        ACC = accuracy_score(y_t, y_c)
        AUC = roc_auc_score(y_t, y_p)
        precision, recall, thresholds = precision_recall_curve(y_t, y_p)
        AUPR = auc(recall, precision)
        BACC = balanced_accuracy_score(y_t, y_c)
        MCC = matthews_corrcoef(y_t, y_c)
        Recall = recall_score(y_t, y_c)
        Precision = precision_score(y_t, y_c)
        Rkcc = compute_RkCC(mcm[c]) 

        res_mcc.append(round(MCC, 3))
        res_acc.append(round(ACC, 3))
        res_auc.append(round(AUC, 3))
        res_aupr.append(round(AUPR, 3))
        res_recall.append(round(Recall, 3))
        res_precision.append(round(Precision, 3))
        res_f1.append(round(F1, 3))
        res_bacc.append(round(BACC, 3))
        res_rkcc.append(round(Rkcc, 3))

    df = pd.DataFrame({'ACC': res_acc, 'BACC': res_bacc,'AUC': res_auc, 'MCC': res_mcc, 'AUPR': res_aupr, 'F1': res_f1,
                       'Precision': res_precision, 'Recall': res_recall, 'Rkcc': res_rkcc}, index=class_names)
    if show:
        print(df)
    if save is not None:
        df.to_csv(save)

    return df


def overall_metrics(y_pred: np.array, y_true: np.array, threshold=0.5, save = None, show = True):
    """
    综合评价多标签分类任务
    """
    y_pred_cls = np.zeros_like(y_pred)
    y_pred_cls[y_pred > threshold] = 1    # 预测类别


    HLoss = hamming_loss(y_true, y_pred_cls)

    # Calculate metrics globally by counting the total true positives,false negatives and false positives.
    F1_micro = f1_score(y_true, y_pred_cls, average='micro')

    # Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.

    F1_macro = f1_score(y_true, y_pred_cls, average='macro')

    # Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
    # This alters 'macro' to account for label imbalance; it can result in an F-score that is not between precision and recall.

    F1_weighted = f1_score(y_true, y_pred_cls, average='weighted')

    df = pd.DataFrame({'HLoss': [HLoss], 'F1_micro': [F1_micro], 'F1_macro': [F1_macro], 'F1_weighted': [F1_weighted]})
    if show:
        print(df)

    if save is not None:
        df.to_csv(save)

    return df


def compute_RkCC(CM):     
    
    '''
    Function to compute the K-category correlation coefficient
    http://www.sciencedirect.com/science/article/pii/S1476927104000799
    
    http://rk.kvl.dk/suite/04022321447260711221/
    
    code : https://github.com/denson/compute_RkCC
    
    Parameters
    ----------
    CM : k X k confusion matrix of int
    
    n_samples : int
    
    
    Returns
    -------
    RkCC: float
    
    '''
    rows, cols = np.shape(CM)
    
    RkCC_numerator=0
    for k_ in range(cols):
        for l_ in range(cols):
            for m_ in range(cols):
    
                this_term = (CM[k_,k_] * CM[m_,l_]) - \
                    (CM[l_,k_] * CM[k_,m_])
    
                RkCC_numerator = RkCC_numerator + this_term
    
    RkCC_denominator_1=0           
    for k_ in range(cols):
        RkCC_den_1_part1=0
        for l_ in range(cols):
            RkCC_den_1_part1= RkCC_den_1_part1+CM[l_,k_]
    
        RkCC_den_1_part2=0
        for f_ in range(cols):
            if f_ != k_:
    
                for g_ in range(cols):
    
                    RkCC_den_1_part2= RkCC_den_1_part2+CM[g_,f_]
    
        RkCC_denominator_1=(RkCC_denominator_1+(RkCC_den_1_part1*RkCC_den_1_part2))
    
    RkCC_denominator_2=0
    for k_ in range(cols):
        RkCC_den_2_part1=0
        for l_ in range(cols):
            RkCC_den_2_part1= RkCC_den_2_part1+CM[k_,l_]
    
        RkCC_den_2_part2=0
        for f_ in range(cols):
            if f_ != k_:
    
                for g_ in range(cols):
    
                    RkCC_den_2_part2= RkCC_den_2_part2+CM[f_,g_]
    
        RkCC_denominator_2=(RkCC_denominator_2+(RkCC_den_2_part1*RkCC_den_2_part2))
    
    RkCC = (RkCC_numerator)/(np.sqrt(RkCC_denominator_1)* np.sqrt(RkCC_denominator_2))
    
    return RkCC




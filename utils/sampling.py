import os
from utils.util_methods import fasta_parser
import numpy as np
import torch

np.random.seed(123)


def random_sampling_balanced(folder_fasta: str,  *fs : str):
    """
    根据某一类多肽(pt)的正样本数量，随机采样相同数量的负样本
    pt: AAP.txt ....
    """

    seq2class = {}  # 将序列映射到类别
    n_class = len(fs)

    for i, fn in enumerate(fs):
        ids, seqs = fasta_parser(os.path.join(folder_fasta, fn))
        seqs = set(seqs)

        for seq in seqs:
            if seq in seq2class.keys():
                seq2class[seq][i] = 1
            else:
                seq2class[seq] = np.zeros(n_class)
                seq2class[seq][i] = 1

    res_seqs = []
    res_labels = []

    for i, fn in enumerate(fs):
        _, pos_seqs = fasta_parser(os.path.join(folder_fasta, fn))

        # construct negative candidate set
        p_set = set(pos_seqs)
        neg_cad_seqs = []
        for j, nfn in enumerate(fs):
            if j == i: continue
            n_set = set(fasta_parser(os.path.join(folder_fasta, nfn))[1])
            neg_cad_seqs.extend(list(n_set - p_set))    # 在n_set中而不再p_set中

        # Negative sampling
        neg_seqs = []
        sample_index = list(range(len(neg_cad_seqs)))
        sample_index = np.random.choice(sample_index, len(pos_seqs), replace=False)
        for idx in sample_index:
            neg_seqs.append(neg_cad_seqs[idx])

        pos_seqs.extend(neg_seqs)
        cur_seqs = pos_seqs
        cur_labels = []
        for seq in cur_seqs:
            cur_labels.append(seq2class[seq])

        res_seqs.append(cur_seqs)
        res_labels.append(np.array(cur_labels))

    return res_seqs, res_labels

class Sampler(torch.utils.data.sampler.Sampler):
    """
    Instance, class, square-root
    """
    def __init__(self, labels, sample_number = None, target = 0 ,method = 'class', lam = 1.0):

        self.method = method
        self.lam = lam
        self.sample_number = len(labels) if sample_number is None else sample_number

        self.num_total = labels.shape[0]
        self.num_class = labels.shape[1]

        self.num_pos = np.sum(labels, axis=0)
        self.labels = labels

        self.indices = list(range(self.num_total))
        self.init_p = np.zeros((2, self.num_class))
        self.init_p[0, :] = self.num_total - self.num_pos
        self.init_p[1, :] = self.num_pos

        self.sample_weights = torch.DoubleTensor(self.get_weights())

        self.target = target

    def get_weights(self):

        if self.method == 'instance':
            weights_class = self.get_instance_balanced_weights(self.init_p)

        elif self.method == 'class':
            weights_class = self.get_class_balanced_weights()

        elif self.method == 'square':
            weights_class = self.get_square_root_weights(self.init_p)

        elif self.method == 'progress':
            weights_class = self.get_progressive_weights(self.init_p)

        else:
            raise NotImplementedError()

        weights_instance = self.weights_map_class2instance(weights_class)

        return weights_instance

    def weights_map_class2instance(self, weights_class):
        weights = np.zeros_like(self.labels, dtype=np.float)
        for i in range(self.num_class):
            weights[:, i][self.labels[:, i] == 0] = weights_class[0, i] / (self.num_total - self.num_pos[i])
            weights[:, i][self.labels[:, i] == 1] = weights_class[1, i] / self.num_pos[i]
        return weights

    def get_instance_balanced_weights(self, init_p):
        return init_p / np.sum(init_p, axis=0)

    def get_class_balanced_weights(self):
        return np.ones((2, self.num_class)) / 2

    def get_square_root_weights(self, init_p):
        t = np.sqrt(init_p)
        return t / np.sum(t, axis=0)

    def get_progressive_weights(self, init_p):
        p_i = self.get_instance_balanced_weights(init_p)
        p_c = self.get_class_balanced_weights()

        return (1 - self.lam ) * p_c + self.lam * p_i

    def set_target(self, idx):
        self.target = idx

    def __len__(self):
        return self.num_total

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.sample_weights[:, self.target], self.sample_number, replacement=True))




if __name__ == '__main__':
    pts = ['AAP.txt', 'ABP.txt', 'ACP.txt', 'AFP.txt', 'AHTP.txt',
           'AIP.txt', 'AMP.txt', 'APP.txt', 'AVP.txt',
           'CCC.txt', 'CPP.txt', 'DDV.txt', 'PBP.txt', 'QSP.txt', 'TXP.txt']

    random_sampling_balanced('../datasets/out90v3/test', *pts)
import os
import pickle as pkl
import numpy as np

def onehot_encoding(seqs):
    residues = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K',
                'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

    encoding_map = np.eye(len(residues))

    residues_map = {}
    for i, r in enumerate(residues):
        residues_map[r] = encoding_map[i]

    res_seqs = []

    for seq in seqs:
        tmp_seq = [residues_map[r] for r in seq]
        res_seqs.append(np.array(tmp_seq))

    return res_seqs
    
def onehot_encoding2(seqs):

    residues = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K',
                'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

    encoding_map = list(range(0, 20))

    residues_map = {}
    for i, r in enumerate(residues):
        residues_map[r] = encoding_map[i]

    res_seqs = []

    for seq in seqs:
        tmp_seq = [residues_map[r] for r in seq]
        res_seqs.append(np.array(tmp_seq).reshape(-1, 1))

    return res_seqs

def kmer_encoding(seqs):
    residues = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K',
                'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

    encoding_map = np.eye(len(residues))
    holder = np.zeros(len(residues))
    residues_map = {}
    for i, r in enumerate(residues):
        residues_map[r] = encoding_map[i]

    res_seqs = []

    for seq in seqs:
        tmp_seq = []
        for i, r in enumerate(seq):
            if i == 0:
                t = []
                t.extend(holder)
                t.extend(residues_map[seq[i]])
                t.extend(residues_map[seq[i + 1]])
                tmp_seq.append(t)

            elif i == len(seq) - 1:
                t = []
                t.extend(residues_map[seq[i - 1]])
                t.extend(residues_map[seq[i]])
                t.extend(holder)
                tmp_seq.append(t)
            else:
                t = []
                t.extend(residues_map[seq[i - 1]])
                t.extend(residues_map[seq[i]])
                t.extend(residues_map[seq[i + 1]])
                tmp_seq.append(t)

        res_seqs.append(np.array(tmp_seq))
    return res_seqs

def _pssm_seq2fn_dict(pssm_folder, save = 'data/pssm_seq2fn.pkl'):
    """
    将已经得到PSSM的文件总结为一个seq:filename的字典，方便根据seq查找pssm文件
    :param folder:
    """
    if pssm_folder[-1] != '/' : pssm_folder += '/'
    fs = os.listdir(pssm_folder)
    res = {}
    for fn in fs:
        with open(pssm_folder + fn , 'r') as f:
            lines = f.readlines()
            tmp = []
            for line in lines[3:]:
                line = line.strip()
                lst = line.split(' ')
                while '' in lst:
                    lst.remove('')
                if len(lst) == 0:
                    break
                r = lst[1]
                tmp.append(r)
            seq = ''.join(tmp)
            res[seq] = fn

    if save is not None:
        with open(save, 'wb') as f:
            pkl.dump(res, f)
    return res

def _msa_seq2fn_dict(msa_folder, save = None):
    """
    将已经得到MSA的文件总结为一个seq:filename的字典，方便根据seq查找msa(a3m)文件
    """
    if msa_folder[-1] != '/' : msa_folder += '/'
    fs = os.listdir(msa_folder)
    res = {}
    for fn in fs:
        with open(msa_folder + fn , 'r') as f:
            lines = f.readlines()
            seq = lines[1].strip()
            res[seq] = fn
    if save is not None:
        with open(save, 'wb') as f:
            pkl.dump(res, f)

    return res

def pssm_encoding(seqs, pssm_dir, blosum = True):
    """
    比对已生成的PSSM矩阵，如果比对失败，检查是否使用blosum, 若是，则用blosum替代，否则为空
    """
    if pssm_dir[-1] != '/': pssm_dir += '/'

    global blosum_dict
    if blosum:
        blosum_dict = _read_blosum('data/blosum62.pkl')

    with open('data/pssm_seq2fn.pkl', 'rb') as f:
        pssm_path_dict = pkl.load(f)

    res = []
    for i , seq in enumerate(seqs):

        if seq in pssm_path_dict.keys():
            # pssm
            pssm_fn = pssm_path_dict[seq]
            tmp = _load_pssm(pssm_fn, pssm_dir)
            res.append(np.array(tmp))
        else:
            if blosum:
                enc = _one_blosum_encoding(seq, blosum_dict)
                res.append(np.array(enc))
            else:
                res.append([])
    return res

def _read_blosum(blosum_dir):
    """Read blosum dict and delete some keys and values."""
    with open(blosum_dir, 'rb') as f:
        blosum_dict = pkl.load(f)

    blosum_dict.pop('*')
    blosum_dict.pop('B')
    blosum_dict.pop('Z')
    blosum_dict.pop('X')
    blosum_dict.pop('alphas')

    for key in blosum_dict:
        for i in range(4):
            blosum_dict[key].pop()
    return blosum_dict

def _load_pssm(query, pssm_path):
    """
    :param query: query id
    :param pssm_path: dir saving pssm files
    :return:
    """
    if pssm_path[-1] != '/': pssm_path += '/'
    with open(pssm_path + query, 'r') as f:
        lines = f.readlines()
        res = []
        for line in lines[3:]:
            line = line.strip()
            lst = line.split(' ')
            while '' in lst:
                lst.remove('')
            if len(lst) == 0:
                break
            r = lst[2:22]
            r = [int(x) for x in r]
            res.append(r)
    return res

def _one_blosum_encoding(seq, blosum_dict):
    """
    :param seq: a single sequence
    :return:
    """
    enc = []
    for aa in seq:
        enc.append(blosum_dict[aa])
    return enc

def position_onhot_encoding(seqs, max_len):
    """
    20 * max_len
    pos * max_len + residue encoding
    A1, A2, ...,A50, R1, R2, ..., R50
    """
    residues = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K',
                'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

    residue2index = {}
    for i, r in enumerate(residues):
        residue2index[r] = i

    res = []
    for seq in seqs:
        seq_enc = []
        for i, r in enumerate(seq):
            enc = max_len * residue2index[r] + i + 1
            seq_enc.append(enc)
        res.append(np.array(seq_enc).reshape(-1, 1))

    return res


#
# if __name__ == '__main__':
#     ids, seqs = fasta_parser('datasets/AMP.txt')
#     res = pssm_encoding(ids, seqs, '../features/pssm/', True)
#     print(res)

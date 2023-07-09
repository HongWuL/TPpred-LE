import numpy as np


def write_fasta(fn, ids, seqs):
    with open(fn, 'w') as f:
        for i in range(len(ids)):
            if ids[i].startswith('>'):
                f.write(ids[i] + '\n')
            else:
                f.write('>' + ids[i] + '\n')
            f.write(seqs[i] + '\n')

def fasta_parser(fn: str):
    # 加载序列数据
    ids = []
    seqs = []
    id = 0

    with open(fn, 'r') as f:
        lines = f.readlines()
        seq_tmp = ""

        for i, line in enumerate(lines):
            line = line.strip()
            if line[0] == '>':
                id = line.replace('|','_')
                id = id.split(' ')[0]
            elif i < len(lines) - 1 and lines[i+1][0] != '>':
                seq_tmp += line.strip()
            else:
                seq_tmp += line.strip()
                seqs.append(seq_tmp)
                ids.append(id)
                id = 0
                seq_tmp = ""

    return ids, seqs

def cat(*args):
    res = args[0]
    for matrix in args[1:]:
        for i in range(len(matrix)):
            res[i] = np.hstack((res[i], matrix[i]))
    return res

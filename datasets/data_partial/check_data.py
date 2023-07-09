from utils.util_methods import fasta_parser, write_fasta
import pandas as pd
import numpy as np

ids_test, seqs_test = fasta_parser('test/seqs.fasta')
ids_val, seqs_val = fasta_parser('val/seqs.fasta')
ids_train, seqs_train = fasta_parser('train/seqs.fasta')

labels_test = np.array(pd.read_csv('test/labels.csv').values)
labels_val = np.array(pd.read_csv('val/labels.csv').values)
labels_train = np.array(pd.read_csv('train/labels.csv').values)

print(labels_test.sum(axis=0))
print(labels_val.sum(axis=0))
print(labels_train.sum(axis=0))

print(labels_test.sum(axis=0) + labels_val.sum(axis=0) + labels_train.sum(axis=0))

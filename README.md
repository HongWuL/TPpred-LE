# TPpred-LE
The implementation of the paper ***TPpred-LE: Therapeutic peptide functions prediction based on label embedding***

## Requirements
The majoy dependencies used in this project are as following:

```
python  3.7
numpy 1.21.6
tqdm  4.64.1
pyyaml  6.0
scikit-learn  1.0.2
torch  1.11.0+cu113
tensorflow  1.14.0
tensorboardX  2.5.1
transformers  4.25.1
```
More detailed python libraries used in this project are referred to `requirements.txt`. 

# Usage
(1) Generate the pssms by blast against NR database(https://ftp.ncbi.nlm.nih.gov/blast/db/).
(2) train and test the model:
Train the model(*Algorithm 1*):
```shell
./train.sh
```
Retrain the model(*Algorithm 2*):
```shell
./retrain.sh
```
The `(re)train_partial.sh` is used to train with the limited datasets.

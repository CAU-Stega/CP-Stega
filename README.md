# Multi-task Few-shot Text Steganalysis based on Context-Attentive Prototypes

This repository contains the code and data for our paper:

*Multi-task Few-shot Text Steganalysis based on Context-Attentive Prototypes*. 

### Quickstart
Run our model with default settings. By default we load data from `stego_data/`.
```
./proto.sh
```
Scripts for other baselines may be found under `bin/`.

## Code
`src/main.py` may be run with one of three modes: `train`, `test`, and `finetune`.
- `train` trains the meta-model using episodes sampled from the training data.
- `test` evaluates the current meta-model on 1000 episodes sampled from the testing data.
- `finetune` trains a fully-supervised classifier on the training data and finetunes it on the support set of each episode, sampled from the testing data.


#### Dependencies
- Python 3.7
- PyTorch 1.1.0
- numpy 1.15.4
- torchtext 0.4.0
- pytorch-transformers 1.1.0
- termcolor 1.1.0
- tqdm 4.32.2



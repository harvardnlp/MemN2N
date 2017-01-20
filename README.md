# End-To-End Memory Network

Torch implementation of MemN2N ([Sukhbaatar, 2015](https://arxiv.org/pdf/1503.08895v5.pdf)). Supports Adjacent Weight Tying, Position Encoding, Temporal Encoding and Linear Start. Code uses v1.0 of [bAbI dataset](https://research.fb.com/projects/babi/) with 1k questions per task.

## Prerequisites:
- Python 2.7
- Torch (with nngraph)

## Preprocessing
First, preprocess included data into hdf5 format:
```
python preprocess.py
```
This will create a hdf5 file for each task (total 20 tasks).

To train:
```
th train.lua
```
This will train on task 16 with Linear Start over 100 epochs by default. See `train.lua` (or the paper) for hyperparameters and more training options. To train on gpu, use `-cuda`.

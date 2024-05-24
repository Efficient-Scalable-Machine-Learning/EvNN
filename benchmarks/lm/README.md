EGRU Language Modeling
===================
To run the language modeling experiments, first download the data

    ./getdata <data_dir>

We [provide checkpoints for EGRU](https://cloudstore.zih.tu-dresden.de/index.php/s/NPQ9pLnpZnTsM5X) with 3 layers of hidden size (1350, 1350, 750)

# Penn Treebank
To train EGRU on Penn Treebank word-level language modeling, run

    python benchmarks/lm/train.py --data=/path/to/data --scratch=/your/scratch/directory/Experiments --dataset=PTB --epochs=1000 --batch_size=64 --rnn_type=egru --layer=3 --bptt=70 --scheduler=cosine --weight_decay=0.10 --learning_rate=0.0012 --learning_rate_thresholds 0.0 --emb_dim=750 --dropout_emb=0.6 --dropout_words=0.1 --dropout_forward=0.25 --grad_clip=0.1 --thr_init_mean=0.01 --dropout_connect=0.7 --hidden_dim=1350 --pseudo_derivative_width=3.6 --scheduler_start=700 --seed=9612

For inference with the [provided checkpoint](https://cloudstore.zih.tu-dresden.de/index.php/s/NPQ9pLnpZnTsM5X), run

    python benchmarks/lm/infer.py --data /path/to/data --dataset PTB --datasplit test --batch_size 1 --directory /path/to/checkpoint

# Wikitext-2
To train EGRU on Wikitext-2, run

    python benchmarks/lm/train.py --data=/your/data/directory --scratch=/your/scratch/directory/Experiments --dataset=WT2 --epochs=800 --batch_size=128 --rnn_type=egru --layer=3 --bptt=70 --scheduler=cosine --weight_decay=0.12 --learning_rate=0.001 --learning_rate_thresholds 0.0 --emb_dim=750 --dropout_emb=0.7 --dropout_words=0.1 --dropout_forward=0.25 --grad_clip=0.1 --thr_init_mean=0.01 --dropout_connect=0.7 --hidden_dim=1350 --pseudo_derivative_width=3.6 --scheduler_start=400 --seed=913420

For inference with the [provided checkpoint](https://cloudstore.zih.tu-dresden.de/index.php/s/NPQ9pLnpZnTsM5X), run

    python benchmarks/lm/infer.py --data /path/to/data --dataset WT2 --datasplit test --batch_size 1 --directory /path/to/checkpoint

Various flags can be passed to change the defaults parameters. 
See "train.py" for a list of all available arguments.

This code was tested with PyTorch >= 1.9.0, CUDA 11.

A large batch of code stems from Salesforce AWD-LSTM implementation:
https://github.com/salesforce/awd-lstm-lm
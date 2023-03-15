EGRU Language Modeling
===================
To run the language modeling experiments, first download the data

    ./getdata <data_dir>

Then run Penn Treebank experiments with EGRU (1350 units)

    python lm/train.py --data path_to_your_data --scratch ./log --dataset PTB --epochs 2500 --rnn_type egru --layers 3 --hidden_dim 1350 --batch_size=64 --bptt=68 --dropout_connect=0.6788113982442464 --dropout_emb=0.7069992062976298 --dropout_forward=0.2641540030663871 --dropout_words=0.05460274136214911 --emb_dim=788 --learning_rate=0.00044406742918918466 --pseudo_derivative_width=2.179414375864446 --thr_init_mean=-3.76855645544185 --weight_decay=9.005509348932795e-06 --seed 12008

or EGRU (2000 units)

    python lm/train.py --data path_to_your_data --scratch ./log --dataset PTB --epochs 2500 --rnn_type egru --layers 3 --hidden_dim 2000 --batch_size=128 --bptt=67 --dropout_connect=0.621405385527356 --dropout_emb=0.7651296208061924 --dropout_forward=0.24131807369801447 --dropout_words=0.14942681962154375 --emb_dim=786 --learning_rate=0.000494172266064804 --pseudo_derivative_width=2.35216907207571 --thr_init_mean=-3.4957794302256007 --weight_decay=6.6878095661652755e-06 --seed 52798

Various flags can be passed to change the defaults parameters. 
See "train.py" for a list of all available arguments.

This code was tested with PyTorch >= 1.9.0

A large batch of code stems from Salesforce AWD-LSTM implementation:
https://github.com/salesforce/awd-lstm-lm
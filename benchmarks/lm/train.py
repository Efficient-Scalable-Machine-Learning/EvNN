# Copyright (c) 2023  Khaleelulla Khan Nazeer, Anand Subramoney, Mark Schöne, David Kappel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import math
import os
import time
from enum import Enum
import numpy as np
import torch
import yaml
from torch import nn

import data as d
from eval import evaluate
from models import LanguageModel


class RNNType(Enum):
    LSTM = 'lstm'
    GRU = 'gru'
    EGRU = 'egru'


def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=16816)
    argparser.add_argument('--data', type=str, required=False, help='path to datasets')
    argparser.add_argument('--dataset', type=str, default='PTB', choices=['WT2', 'PTB', 'WT103'])
    argparser.add_argument('--scratch', type=str, default='./scratch', help='scratch directory for jobs')
    argparser.add_argument('--epochs', type=int, default=800)
    argparser.add_argument('--batch_size', type=int, default=80)
    argparser.add_argument('--learning_rate', type=float, default=0.0003)
    argparser.add_argument('--learning_rate_thresholds', type=float, default=1.0)
    argparser.add_argument('--bptt', type=int, default=70)
    argparser.add_argument('--grad_clip', type=float, default=2.0)
    argparser.add_argument('--rnn_type', type=str,
                           default='egru', choices=[e.value for e in RNNType])
    argparser.add_argument('--hidden_dim', type=int, default=1350)
    argparser.add_argument('--layers', type=int, default=3)
    argparser.add_argument('--emb_dim', type=int, default=400)
    argparser.add_argument('--projection', action='store_true')
    argparser.add_argument('--dropout_emb', type=float, default=0.4)
    argparser.add_argument('--dropout_words', type=float, default=0.1)
    argparser.add_argument('--dropout_forward', type=float, default=0.3)
    argparser.add_argument('--dropout_connect', type=float, default=0.5)
    argparser.add_argument('--checkpoint', type=str, required=False, default="")
    argparser.add_argument('--log_interval', type=int, default=1000)
    argparser.add_argument('--scheduler', type=str, default='lambda', choices=['lambda', 'cosine', 'step'])
    argparser.add_argument('--scheduler_start', type=int, default=200)
    argparser.add_argument('--momentum', type=float, default=0.0)
    argparser.add_argument('--weight_decay', type=float, default=1.2e-6)
    argparser.add_argument('--alpha', type=float, default=0,
                           help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    argparser.add_argument('--beta', type=float, default=0,
                           help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
    argparser.add_argument('--gamma', type=float, default=0,
                           help='EGRU activity regularization')
    argparser.add_argument('--pseudo_derivative_width', type=float, default=1.0)
    argparser.add_argument('--thr_init_mean', type=float, default=0.2)
    argparser.add_argument('--weight_init_gain', type=float, default=1.0)
    argparser.add_argument('--prune', nargs='*', type=float, default=0.0)

    return argparser.parse_args()


def main(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # load dataset
    train_data, val_data, test_data, vocab_size = d.get_data(
        root=args.data,
        dset=args.dataset,
        batch_size=args.batch_size,
        device=device
    )
    print(f"Dataset {args.dataset} has {vocab_size} tokens")

    criterion = nn.CrossEntropyLoss()

    # load the model
    model_args = {
        'rnn_type'         : args.rnn_type,
        'nlayers'          : args.layers,
        'projection'       : args.projection,
        'emb_dim'          : args.emb_dim,
        'hidden_dim'       : args.hidden_dim,
        'vocab_size'       : vocab_size,
        'dropout_words'    : args.dropout_words,
        'dropout_embedding': args.dropout_emb,
        'dropout_connect'  : args.dropout_connect,
        'dropout_forward'  : args.dropout_forward,
        'alpha'            : args.alpha,
        'beta'             : args.beta,
        'gamma'            : args.gamma
    }

    if args.rnn_type == 'lstm' or args.rnn_type == 'gru':
        model = LanguageModel(**model_args)
    elif args.rnn_type == 'egru':
        model = LanguageModel(
            **model_args,
            dampening_factor=args.pseudo_derivative_width,
            pseudo_derivative_support=args.pseudo_derivative_width,
            grad_clip=args.grad_clip,
            thr_mean=args.thr_init_mean,
            weight_initialization_gain=args.weight_init_gain
        )
    else:
        raise RuntimeError("Unknown RNN type: %s" % args.rnn_type)
    print("RNN parameters: ", list(map(lambda x: x[0], model.named_parameters())))

    if len(args.checkpoint) > 0:
        model.load_state_dict(torch.load(args.checkpoint))
    model = model.to(device)
    # get the dimensions of the hidden state
    if args.rnn_type == 'egru':
        hidden_dims = [rnn.hidden_size for rnn in model.rnns]
    else:
        hidden_dims = [rnn.module.hidden_size if args.dropout_connect > 0 else rnn.hidden_size for rnn in model.rnns]

    config = vars(args)
    config.update({'num_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)})
    config.update({'SLURM_JOB_ID': os.getenv('SLURM_JOB_ID')})
    print(f"Model Parameter Count: {config['num_parameters']}")

    model_signature = 'rnn_type={0}__nlayers{1}_lr={2}_decay={3}'.format(args.rnn_type, args.layers, args.learning_rate, args.weight_decay)

    # MODEL PRUNING
    pruning = False
    print(args.prune)
    if isinstance(args.prune, list) and len(args.prune) == 1:
        pruning = True
        args.prune = args.prune[0]
    elif isinstance(args.prune, list) and len(args.prune) == args.layers:
        pruning = True

    if pruning:
        prune(
            model=model,
            criterion=criterion,
            data=train_data,
            batch_size=args.batch_size,
            sequence_length=args.bptt,
            ntokens=vocab_size,
            device=device,
            hidden_dims=hidden_dims,
            fractions=args.prune
        )

    return_bw_sparsity = True if model.rnn_type == 'egru' else False

    config = vars(args)
    config.update({'num_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)})
    print(f"Model Parameter Count: {config['num_parameters']}")

    # setup training
    param_groups = [
        # most parameters
        {'params': [param for name, param in model.named_parameters()
                    if 'thr' not in name and 'layernorm' not in name]},
        # layernorm
        {'params': [param for name, param in model.named_parameters()
                    if 'layernorm' in name],
         'weight_decay': 0},
        # thresholds
        {'params': [param for name, param in model.named_parameters()
                    if 'thr' in name],
         'lr': args.learning_rate * args.learning_rate_thresholds,
         'weight_decay': 0}
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=args.weight_decay, betas=(args.momentum, 0.999))
    
    if args.scheduler == "lambda":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda ep: 1.0)
    
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs-args.scheduler_start), eta_min=0)

    if args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(args.epochs-args.scheduler_start)/4, gamma=args.gamma)

    best_val_loss = float('inf')

    output_path = os.path.join(args.scratch, args.dataset, args.rnn_type.upper(),
                               f"{model_signature}_{time.strftime('%y-%m-%d-%H:%M:%S')}")
    best_model_path = os.path.join(output_path,
                                   'checkpoints', f'{args.rnn_type.upper()}_best_model.cpt')
    print("Saving model weights to", best_model_path)

    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    with open(os.path.join(output_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    n = 0
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        train_results = train(
            model=model,
            train_data=train_data,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
            batch_size=args.batch_size,
            bptt=args.bptt,
            ntokens=vocab_size,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            device=device,
            return_backward_sparsity=return_bw_sparsity
        )

        if return_bw_sparsity:
            train_loss, bw_sparsity = train_results
        else:
            train_loss = train_results

        val_loss, mean_activity, layerwise_activity_mean, layerwise_activity_std, centered_cell_states = \
            evaluate(
                model=model,
                eval_data=val_data,
                criterion=criterion,
                batch_size=args.batch_size,
                bptt=args.bptt,
                ntokens=vocab_size,
                hidden_dims=hidden_dims,
                device=device
            )

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_model(model.state_dict(), best_model_path)
            n = 0
        else:
            n += 1

        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'train loss {train_loss:5.2f} | '
              f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f} | mean activity {mean_activity:.4f}')

        if return_bw_sparsity:
            mean_bw_sparsity = np.dot(bw_sparsity, np.array(hidden_dims)) / np.sum(np.array(hidden_dims))
            print(f'backward sparsity {mean_bw_sparsity}')
        print('-' * 89)

        # if the loss diverged to infinity, stop training
        if np.isnan(val_loss).any():
            print(f"EXITING DUE TO NAN LOSS {val_loss}")
            break

        if epoch > args.scheduler_start:
            scheduler.step()

    ######################################################################
    # Evaluate the best model on the test dataset
    # -------------------------------------------
    #
    test_batch_size = 1
    train_data, val_data, test_data, vocab_size = d.get_data(root=args.data,
                                                             dset=args.dataset,
                                                             batch_size=test_batch_size,
                                                             device=device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_activity, test_layerwise_activity_mean, test_layerwise_activity_std, centered_cell_states = \
        evaluate(
            model=model,
            eval_data=test_data,
            criterion=criterion,
            batch_size=test_batch_size,
            bptt=args.bptt,
            ntokens=vocab_size,
            hidden_dims=hidden_dims,
            device=device
        )

    test_ppl = math.exp(test_loss)
    print('=' * 89)
    print(f'| End of training | test loss {test_loss:5.2f} | '
          f'test ppl {test_ppl:8.2f} | '
          f'test mean activity {test_activity}')
    print(f'Layerwise activity {test_layerwise_activity_mean} +- {test_layerwise_activity_std}')
    print('=' * 89)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if h is not None:
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)


def prune(model, criterion, data, batch_size, sequence_length, ntokens, device, hidden_dims, fractions=0.0):
    print("Pruning model...")
    test_loss, test_activity, test_layerwise_activity_mean, test_layerwise_activity_std, centered_cell_states, all_hiddens = \
        evaluate(
            model=model,
            eval_data=data,
            criterion=criterion,
            batch_size=batch_size,
            bptt=sequence_length,
            ntokens=ntokens,
            device=device,
            hidden_dims=hidden_dims,
            return_hidden=True
        )

    model.prune(fractions, all_hiddens, device)
    print(f"Perplexity before pruning {math.exp(test_loss)}")


def train(model,
          train_data,
          optimizer,
          criterion,
          epoch,
          batch_size,
          bptt,
          ntokens,
          grad_clip,
          log_interval,
          device,
          return_backward_sparsity=False):
    model.train()  # turn on train mode
    total_loss = 0.
    cur_loss = 0.
    start_time = time.time()
    hidden = model.init_hidden(batch_size)

    # log backward sparsity
    bw_sparsity_layers = []

    num_batches = len(train_data) // bptt
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:

        # Random Backpropagation through time (BPTT) sequence length
        this_bptt = bptt if np.random.random() < 0.95 else bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(this_bptt, 5)))

        # adjust learning rate for sequence length
        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / this_bptt

        # fetch data and make it batch first
        data, targets = d.get_batch(train_data, i, seq_len=seq_len, batch_first=True)

        # prepare forward pass
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        # forward pass
        output, hidden, hid_full, hid_dropped = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)

        total_loss += loss.item() * data.numel()
        cur_loss += loss.item()

        loss += model.ar_loss

        # backward pass and weight update
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # output monitoring
        output_gate_vals = [torch.where(hid == 0, torch.zeros_like(hid), torch.ones_like(hid)).to(device) for hid in
                            hid_full]
        mean_activity = torch.mean(torch.cat([ogv.flatten() for ogv in output_gate_vals]))
        if return_backward_sparsity:
            bw_sparsity_layers.append(model.backward_sparsity)

        optimizer.param_groups[0]['lr'] = lr2
        if batch % log_interval == 0 and batch > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = cur_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr2:02.4f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f} | mean activity {mean_activity:.4f} ')
            if return_backward_sparsity:
                print(f'backward sparsity {bw_sparsity_layers[-1]}')
            start_time = time.time()
            cur_loss = 0.

        batch += 1
        i += seq_len
    if return_backward_sparsity:
        return total_loss / train_data.numel(), torch.mean(torch.stack(bw_sparsity_layers), dim=0)
    else:
        return total_loss / train_data.numel()


def checkpoint_model(state_dict, path):
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save(state_dict, path)
    print(f'saving model as {path}')


if __name__ == "__main__":
    args = get_args()
    main(args)

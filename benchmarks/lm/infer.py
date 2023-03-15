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
import os
import math
import numpy as np
import yaml
import h5py

import torch
import torch.nn

import lm.data as d
from lm.models import LanguageModel
from lm.eval import evaluate


def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=3000)
    argparser.add_argument('--data', type=str, required=True, help='path to datasets')
    argparser.add_argument('--dataset', type=str, default='PTB', choices=['WT2', 'PTB'])
    argparser.add_argument('--datasplit', type=str, default='val', choices=['train', 'val', 'test'])
    argparser.add_argument('--batch_size', type=int, default=80)
    argparser.add_argument('--directory', type=str, required=False, help='model directory for checkpoints and config')
    argparser.add_argument('--hidden', action='store_true', help='returns the hidden states of the whole dataset to perform analysis')
    argparser.add_argument('--prune', type=float, default=0.0)

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
    train_data, val_data, test_data, vocab_size = d.get_data(root=args.data,
                                                             dset=args.dataset,
                                                             batch_size=args.batch_size,
                                                             device=device)

    test_data = {'test': test_data, 'val': val_data, 'train': train_data}[args.datasplit]

    with open(os.path.join(args.directory, 'config.yaml'), 'r') as file:
        #file.readline()
        #file.readline()
        config = yaml.load(file, Loader=yaml.Loader)

    # load the model
    model_args = {
        'rnn_type': config['rnn_type'],
        'nlayers': config['layers'],
        'projection': config['projection'],
        'emb_dim': config['emb_dim'],
        'hidden_dim': config['hidden_dim'],
        'vocab_size': vocab_size,
        'dropout_words': config['dropout_words'],
        'dropout_embedding': config['dropout_emb'],
        'dropout_connect': config['dropout_connect'],
        'dropout_forward': config['dropout_forward'],
        'alpha': config['alpha'],
        'beta': config['beta'],
        'gamma': config['gamma']}

    if config['rnn_type'] == 'lstm' or config['rnn_type'] == 'gru':
        model = LanguageModel(**model_args).to(device)
    elif config['rnn_type'] == 'egru':
        model = LanguageModel(**model_args,
                              dampening_factor=config['damp_factor'],
                              pseudo_derivative_support=config['pseudo_derivative_width']).to(device)
    else:
        raise RuntimeError("Unknown RNN type: %s" % config['rnn_type'])

    best_model_path = os.path.join(args.directory, 'checkpoints', f"{config['rnn_type'].upper()}_best_model.cpt")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    criterion = torch.nn.CrossEntropyLoss()

    if args.hidden:
        test_loss, test_activity, test_layerwise_activity_mean, test_layerwise_activity_std, centered_cell_states, all_hiddens = \
            evaluate(model=model,
                     eval_data=test_data,
                     criterion=criterion,
                     batch_size=args.batch_size,
                     bptt=config['bptt'],
                     ntokens=vocab_size,
                     device=device,
                     return_hidden=True)
        save_file = os.path.join(args.directory, f'hidden_states_{args.datasplit}.hdf')
        with h5py.File(save_file, 'w') as f:
            print(f'Writing hidden states to {save_file}')
            for i, hid in enumerate(all_hiddens):
                f.create_dataset(f'hidden_states_{i}', data=hid.cpu().numpy())
                if config['rnn_type'] == 'egrud':
                    f.create_dataset(f'centered_cell_states_{i}', data=centered_cell_states[i].cpu().numpy())
    else:
        test_loss, test_activity, test_layerwise_activity_mean, test_layerwise_activity_std, centered_cell_states = \
            evaluate(model=model,
                     eval_data=test_data,
                     criterion=criterion,
                     batch_size=args.batch_size,
                     bptt=config['bptt'],
                     ntokens=vocab_size,
                     device=device,
                     return_hidden=False)

    test_ppl = math.exp(test_loss)
    print('=' * 89)
    print(f'| Inference | test loss {test_loss:5.2f} | '
          f'test ppl {test_ppl:8.2f} | '
          f'test mean activity {test_activity}')
    print(f'Layerwise activity {test_layerwise_activity_mean.tolist()} +- {test_layerwise_activity_std.tolist()}')
    print('=' * 89)

    if args.prune > 0.0 and args.hidden:
        print(f"Model Parameter Count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        input_indices = torch.arange(model.rnns[0].input_size).to(device)
        for i in range(model.nlayers):
            if i < model.nlayers - 1:
                # get event frequencies
                hid_dim = all_hiddens[i].shape[2]
                hid_cells = all_hiddens[i].reshape(-1, hid_dim)
                seq_len = hid_cells.shape[0]
                spike_frequency = torch.sum(hid_cells != 0, dim=0) / seq_len
                print(
                    f"Layer {i + 1}: "
                    f"less than 1/100: {torch.sum(spike_frequency < 0.01)} / {spike_frequency.shape} "
                    f"// never: {torch.sum(hid_cells.sum(dim=0) == 0)} / {spike_frequency.shape}")

                # compute remaining indicies from spike frequencies
                topk = int(model.rnns[i].hidden_size * (1 - args.prune))
                hidden_indices, _ = torch.sort(torch.argsort(spike_frequency, descending=True)[:topk], descending=False)
                hidden_indices = hidden_indices.to(device)
            else:
                hidden_indices = torch.arange(model.rnns[i].hidden_size).to(device)
            model.rnns[i].prune_units(input_indices, hidden_indices)
            input_indices = hidden_indices

        print(f"Model Parameter Count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        test_loss, test_activity, test_layerwise_activity_mean, test_layerwise_activity_std, centered_cell_states, all_hiddens = \
            evaluate(model=model,
                     eval_data=test_data,
                     criterion=criterion,
                     batch_size=args.batch_size,
                     bptt=config['bptt'],
                     ntokens=vocab_size,
                     device=device,
                     return_hidden=True)
        for i in range(model.nlayers - 1):
            # get event frequencies
            hid_dim = all_hiddens[i].shape[2]
            hid_cells = all_hiddens[i].reshape(-1, hid_dim)
            seq_len = hid_cells.shape[0]
            spike_frequency = torch.sum(hid_cells != 0, dim=0) / seq_len
            print(
                f"less than 1/100: {torch.sum(spike_frequency < 0.01)} / {spike_frequency.shape} "
                f"// never: {torch.sum(hid_cells.sum(dim=0) == 0)} / {spike_frequency.shape}")
        test_ppl = math.exp(test_loss)
        print('=' * 89)
        print(f'| Inference | test loss {test_loss:5.2f} | '
              f'test ppl {test_ppl:8.2f} | '
              f'test mean activity {test_activity}')
        print(f'Layerwise activity {test_layerwise_activity_mean.tolist()} +- {test_layerwise_activity_std.tolist()}')
        print('=' * 89)


if __name__ == "__main__":
    args = get_args()
    main(args)

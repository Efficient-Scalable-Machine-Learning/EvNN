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

import os
import argparse
import math
from yaml import Loader

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm
from sdict import sdict
from simmanager import Paths
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from egru import Timer, get_random_name, RNNType
from egru.models import RNNReadoutWrapper, RNNWrapper, RNNMultiLayerWrapper

import haste_pytorch as haste
import evnn_pytorch as evnn


def bitreversal_po2(n):
    m = int(math.log(n) / math.log(2))
    perm = np.arange(n).reshape(n, 1)
    for i in range(m):
        n1 = perm.shape[0] // 2
        perm = np.hstack((perm[:n1], perm[n1:]))
    return perm.squeeze(0)


def bitreversal_permutation(n):
    m = int(math.ceil(math.log(n) / math.log(2)))
    N = 1 << m
    perm = bitreversal_po2(N)
    return np.extract(perm < n, perm)


def sequential_MNIST(batch_size, cuda, data_path, permute=False):
    validation_test_batch_size = batch_size
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    transforms_list = [
        transforms.ToTensor(),
        # transforms.Resize(size=(14, 14)),
        transforms.Lambda(lambda x: x.view(-1, 1))
    ]

    if permute:
        print("Permuting pixels!")
        permutation = bitreversal_permutation(28 * 28)
        transforms_list.append(transforms.Lambda(lambda x: x[permutation]))

    full_train_set = datasets.MNIST(
        data_path, train=True, download=True, transform=transforms.Compose(transforms_list))

    # use 20% of training data for validation
    train_set_size = int(len(full_train_set) * 0.8)
    valid_set_size = len(full_train_set) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, val_set = torch.utils.data.random_split(
        full_train_set, [train_set_size, valid_set_size], generator=seed)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, **kwargs)
    validation_loader = torch.utils.data.DataLoader(val_set, batch_size=validation_test_batch_size, shuffle=True,
                                                    # drop_last=True,
                                                    **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_path, train=False,
                       transform=transforms.Compose(transforms_list)),
        batch_size=validation_test_batch_size, shuffle=False,
        # drop_last=True,
        **kwargs)

    return train_loader, validation_loader, test_loader


def train_batch(model, loss_function, optimizer, ext_inputs, targets, train_params, c, device):
    rnn_out, states = model(ext_inputs)

    if c.rnn_type == RNNType. EGRU:
        c_vals, o_vals, tr_vals = states
        mean_activity = torch.mean(o_vals).to(device)
    else:
        c_vals = rnn_out
        mean_activity = torch.mean(
            torch.where(torch.isclose(rnn_out, torch.tensor(0.)), torch.zeros_like(rnn_out), torch.ones_like(rnn_out)))

    # `model` here is the last model
    if c.use_output_trace:
        relevant_outputs = model.hidden2out(tr_vals)[:, -1, :]
    else:
        relevant_outputs = model.hidden2out(rnn_out)[:, -1, :]

    loss = loss_function(relevant_outputs, targets).to(device)

    total_loss = loss
    if c.activity_regularization:  # and it > 200:
        activity_reg_loss = (
            mean_activity - c.activity_regularization_target) ** 2
        total_loss = total_loss + c.activity_regularization_constant * activity_reg_loss
    if c.voltage_regularization:
        # print("Doing voltage regularization")
        voltage_reg_loss = torch.mean(
            (c_vals - c.voltage_regularization_target) ** 2)
        total_loss = total_loss + c.voltage_regularization_constant * voltage_reg_loss

    optimizer.zero_grad()
    model.zero_grad()
    total_loss.backward()

    total_norm = None
    if c.use_grad_clipping:
        total_norm = torch.nn.utils.clip_grad_norm_(
            train_params, c.grad_clip_norm)

    optimizer.step()

    actual_output = torch.argmax(relevant_outputs, -1)
    success_rate = (actual_output == targets).float().mean()

    return loss, success_rate, mean_activity, total_norm, c_vals


def eval_batch(model, loss_function, ext_inputs, targets, c, device):
    torch.cuda.empty_cache()
    rnn_out, states = model(ext_inputs)

    if c.rnn_type == RNNType. EGRU:
        c_vals, o_vals, tr_vals = states
        mean_activity_ = torch.mean(o_vals).to(device)
    else:
        mean_activity_ = torch.mean(
            torch.where(torch.isclose(rnn_out, torch.tensor(0.)), torch.zeros_like(rnn_out), torch.ones_like(rnn_out)))

    if c.use_output_trace:
        relevant_outputs = model.hidden2out(tr_vals)[:, -1, :]
    else:
        relevant_outputs = model.hidden2out(rnn_out)[:, -1, :]

    loss_ = loss_function(relevant_outputs, targets).to(device)

    actual_output = torch.argmax(relevant_outputs, -1)
    success_rate_ = (actual_output == targets).float().mean()

    return loss_, success_rate_, mean_activity_


def get_rnn(input_size, c):
    if c.rnn_type == RNNType.LSTM:
        assert c.unit_size == 1
        rnn = haste.LSTM(input_size, c.n_units * c.unit_size, dropout=c.dropout_connect, zoneout=c.zoneout,
                         batch_first=True)
        if c.custom_lstm_init:
            print("Doing custom LSTM init")
            for name, param in rnn.named_parameters():
                if 'bias' in name:
                    nn.init.ones_(
                        param[c.n_units * c.unit_size:2 * c.n_units * c.unit_size])
                elif 'weight' in name:
                    nn.init.xavier_uniform_(param[:c.n_units * c.unit_size])
                    nn.init.xavier_uniform_(
                        param[c.n_units * c.unit_size:2 * c.n_units * c.unit_size])
                    nn.init.xavier_uniform_(
                        param[2 * c.n_units * c.unit_size:3 * c.n_units * c.unit_size])
                    nn.init.xavier_uniform_(
                        param[3 * c.n_units * c.unit_size:4 * c.n_units * c.unit_size])
    elif c.rnn_type == RNNType.GRU:
        assert c.unit_size == 1
        rnn = haste.GRU(input_size, c.n_units * c.unit_size, dropout=c.dropout_connect, zoneout=c.zoneout,
                        batch_first=True)
    elif c.rnn_type == RNNType. EGRU:
        rnn = evnn.EGRU(input_size, c.n_units, dropout=c.dropout_connect, zoneout=c.zoneout, batch_first=True,
                        use_custom_cuda=True)
        if c.use_grad_clipping:
            rnn.grad_clip_norm(enable=True, norm=c.grad_clip_norm)
        rnn = RNNWrapper(rnn)
    else:
        raise RuntimeError("Unknown lstm type: %s" % c.rnn_type)
    print(f"{c.rnn_type} parameters: ", list(
        map(lambda x: x[0], rnn.named_parameters())))
    return rnn


def main(c, pt):
    print('Seed: ', c.seed)

    torch.manual_seed(c.seed)
    np.random.seed(c.seed)

    training_data, validation_data, testing_data = sequential_MNIST(c.batch_size, cuda=c.cuda, data_path=data_path,
                                                                    permute=c.permute)

    device = torch.device("cpu")
    if c.cuda:
        device = torch.device("cuda:0")

    INPUT_SIZE = 1
    OUTPUT_SIZE = 10

    model_layers = []

    for l in range(c.n_layers):
        if l == 0:
            input_size = INPUT_SIZE
        else:
            input_size = c.n_units
        rnn = get_rnn(input_size, c)

        model_layers.append(rnn)

    # NOTE: The linear readout is not applied implicitly. It has to be applied outside.
    model = RNNReadoutWrapper(RNNMultiLayerWrapper(model_layers, dropout_forward=c.dropout_forward),
                              output_size=OUTPUT_SIZE)

    resume_epoch = 0
    if c.resume:
        print(f"LOADING MODEL from {c.resume}")
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(c.resume, map_location=torch.device('cpu')),
                                                              strict=False)
        print(
            f"While loading the following keys were missing: {missing_keys}. The following keys were unexpected: {unexpected_keys}")
        s = c.resume
        resume_epoch = int(s.split('.')[0].split('-')[-1])

    model = model.to(device)

    train_params = list(model.parameters())

    if c.use_rmsprop:
        print('Using RMSprop.')
        optimizer = optim.RMSprop(
            train_params, lr=c.learning_rate, weight_decay=0.9)
    else:
        optimizer = optim.Adam(train_params, lr=c.learning_rate)

    scheduler = None
    if c.learning_rate_decay:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=100, gamma=0.8)

    loss_function = nn.CrossEntropyLoss()

    running_avg_success_rate = 0.

    for epoch in range(resume_epoch, c.n_training_epochs):
        torch.cuda.empty_cache()
        if pt:
            torch.save(model.state_dict(), os.path.join(
                pt.results_path, f'models/egurc-{epoch}.pt'))

        training_data = tqdm(training_data)
        model.train()
        for current_batch, (inputs, targets) in enumerate(training_data):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # TRAINING
            with Timer() as bt:
                loss, success_rate, mean_activity, total_norm, c_vals = train_batch(model, loss_function, optimizer,
                                                                                    inputs, targets,
                                                                                    train_params, c, device)
            # END with timer
            running_avg_success_rate += success_rate.data.item()
            running_avg_success_rate /= 2
            loss_ = loss.data.item()
            mean_activity_ = mean_activity.data.item()
            success_rate_ = success_rate.data.item()
            if current_batch % 100 == 0 or current_batch == 1 or current_batch == len(training_data) - 1:
                print(f"Training epoch {epoch}, batch {current_batch} :: Loss is {loss_:.4f} :: Success rate"
                      f" {success_rate_:.4f} (Running avg.  {running_avg_success_rate:.4f}) ::"
                      f" Mean activity {mean_activity_:.4f} :: "
                      f" Batch time was {bt.difftime:.4f}.")
                if c.use_grad_clipping:
                    print(
                        f'Total norm of gradients before clipping: {total_norm:.4f}')

        # END for current_batch
        if scheduler:
            scheduler.step()

        # VALIDATION
        model.eval()
        with torch.no_grad():
            loss, success_rate = 0., 0.
            validation_data = tqdm(validation_data)
            for current_batch, (inputs, targets) in enumerate(validation_data):
                torch.cuda.empty_cache()
                inputs = inputs.to(device)
                targets = targets.to(device)

                loss_, success_rate_, mean_activity_ = eval_batch(model, loss_function, inputs, targets, c,
                                                                  device)
                mean_activity += mean_activity_
                loss += loss_
                success_rate += success_rate_

                if current_batch % 100 == 0 or current_batch == 1 or current_batch == len(training_data) - 1:
                    print(f"Validation epoch {epoch}, batch {current_batch} :: Loss is {loss_:.4f} :: Success rate"
                          f" {success_rate_:.4f} ::"
                          f" Mean activity {mean_activity_:.4f}")

            # END current_batch
        # END with torch.nograd()
        loss_ = loss.data.item() / (current_batch + 1)
        mean_activity_ = mean_activity.data.item() / (current_batch + 1)
        success_rate_ = success_rate.data.item() / (current_batch + 1)

        print(f"Validation in {epoch} :: Loss is {loss_:.4f} :: "
              f" Mean activity {mean_activity_:.4f} :: "
              f"Success rate {success_rate_:.4f} :: Batch time was {bt.difftime:.4f}.")

        if epoch % 100 == 0 or epoch == c.n_training_epochs - 1:
            # TEST
            model.eval()
            with torch.no_grad():
                loss, success_rate = 0., 0.
                if c.location not in ['jusuf', 'taurus']:
                    testing_data = tqdm(testing_data)
                for current_batch, (inputs, targets) in enumerate(testing_data):
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    loss_, success_rate_, mean_activity_ = eval_batch(model, loss_function, inputs, targets, c,
                                                                      device)
                    loss += loss_
                    success_rate += success_rate_

                    if current_batch % 10 == 0 or current_batch == 1 or current_batch == len(training_data) - 1:
                        print(f"Test batch {current_batch} :: Loss is {loss_:.4f} :: Success rate"
                              f" {success_rate_:.4f} ::"
                              f" Mean activity {mean_activity_:.4f}")
                        if c.use_grad_clipping:
                            print(
                                f'Total norm of gradients before clipping: {total_norm:.4f}')

                # END current_batch
            # END with torch.nograd()
            loss_ = loss.data.item() / (current_batch + 1)
            mean_activity_ = mean_activity.data.item() / (current_batch + 1)
            success_rate_ = success_rate.data.item() / (current_batch + 1)

            print(f"Test :: Loss is {loss_:.4f} :: "
                  f"Mean activity {mean_activity_:.4f} :: "
                  f"Success rate {success_rate_:.4f} :: Batch time was {bt.difftime:.4f}.")
            # End TEST
    # END for epoch


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=3000)
    argparser.add_argument('--resume', type=str)

    argparser.add_argument('--permute', action='store_true')

    argparser.add_argument('--batch-size', type=int, default=256)
    argparser.add_argument('--learning-rate', type=float, default=0.001)
    argparser.add_argument('--zoneout', type=float, default=0.)
    argparser.add_argument('--dropout-connect', type=float, default=0.)
    argparser.add_argument('--dropout-forward', type=float, default=0.)
    argparser.add_argument('--use-rmsprop', action='store_true')
    argparser.add_argument('--learning-rate-decay', action='store_true')
    argparser.add_argument('--use-grad-clipping', action='store_true')
    argparser.add_argument('--grad-clip-norm', type=float, default=2.0)
    argparser.add_argument('--cuda', action='store_true')
    argparser.add_argument('--use-output-trace', action='store_true')
    argparser.add_argument('--custom-lstm-init', action='store_true')
    argparser.add_argument('--rnn-type', type=str,
                           default='lstm', choices=[e.value for e in RNNType])
    argparser.add_argument('--units', type=int, default=256)
    argparser.add_argument('--layers', type=int, default=1)
    argparser.add_argument('--train-epochs', type=int, default=200)

    argparser.add_argument('--voltage-regularization', action='store_true')
    argparser.add_argument(
        '--voltage-regularization-constant', type=float, default=1.)
    argparser.add_argument(
        '--voltage-regularization-target', type=float, default=-0.9)

    argparser.add_argument('--activity-regularization', action='store_true')
    argparser.add_argument(
        '--activity-regularization-constant', type=float, default=1.)
    argparser.add_argument(
        '--activity-regularization-target', type=float, default=0.05)

    argparser.add_argument('--pseudo-derivative-width', type=float, default=1.)

    argparser.add_argument('--lstm-const-forget-gate', action='store_true')
    argparser.add_argument('--debug', action='store_true')
    argparser.add_argument('--nostore', action='store_true',
                           help='Nothing is stored on disk')
    args = argparser.parse_args()

    if torch.cuda.is_available():
        if not args.cuda:
            print(
                "WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    resume = args.resume

    # START CONFIG

    def get_config():
        print('Generating dictionary of parameters')
        # General
        seed = args.seed

        # (LSTM) Network parameters
        n_training_epochs = args.train_epochs

        # Convert string argument to enum
        for e in RNNType:
            if args.rnn_type == e.value:
                rnn_type = e
                break
        else:
            raise RuntimeError(f"Unknown value {args.rnn_type}")

        if rnn_type in [RNNType.LSTM, RNNType.GRU, RNNType. EGRU]:
            n_units = args.units
        else:
            raise RuntimeError(f"Unknown RNN type {rnn_type}")

        if args.activity_regularization:
            print("using activity regularization")

        batch_size = args.batch_size
        if args.debug:
            print("!!DEBUG!!")
            n_training_epochs = 2
            # batch_size = 10

        config = dict(
            seed=seed,
            cuda=args.cuda,
            resume=resume,
            permute=args.permute,
            n_training_epochs=n_training_epochs,
            rnn_type=rnn_type,
            batch_size=batch_size,
            n_units=n_units,
            n_layers=args.layers,
            use_output_trace=args.use_output_trace,
            custom_lstm_init=args.custom_lstm_init,

            learning_rate=args.learning_rate,
            learning_rate_decay=args.learning_rate_decay,
            zoneout=args.zoneout,
            dropout_connect=args.dropout_connect,
            dropout_forward=args.dropout_forward,
            use_rmsprop=args.use_rmsprop,
            use_grad_clipping=args.use_grad_clipping,
            grad_clip_norm=args.grad_clip_norm,
            activity_regularization=args.activity_regularization,
            activity_regularization_constant=args.activity_regularization_constant,
            activity_regularization_target=args.activity_regularization_target,
            voltage_regularization=args.voltage_regularization,
            voltage_regularization_constant=args.voltage_regularization_constant,
            voltage_regularization_target=args.voltage_regularization_target,
            pseudo_derivative_width=args.pseudo_derivative_width,
        )
        return config

    config = get_config()
    if resume:
        print(f"Loading config from {resume}")
        with open(os.path.join(os.path.dirname(resume), '../..', 'data', 'config.yaml'), 'r') as f:
            loaded_config = yaml.load(f, Loader=Loader)
            for k, v in loaded_config.items():
                if not k in ['cuda', 'debug', 'resume']:
                    config[k] = v
        if args.train_epochs > config['n_training_epochs']:
            config['n_training_epochs'] = args.train_epochs

    print(config)
    config = sdict(config)
    # END CONFIG

    rroot = os.path.expanduser(os.path.join('~', 'output'))
    data_path = './data'

    if args.nostore:  # Needs data_path
        print("!!NOT STORING ANY DATA ON DISK!!")
        if args.debug:
            from ipdb import launch_ipdb_on_exception

            with launch_ipdb_on_exception():
                main(config, None)
        else:
            main(config, None)
    else:
        print(rroot)
        root_dir = os.path.join(rroot, 'egru')
        if args.debug:
            root_dir = os.path.join(rroot, 'tmp')  # NOTE: DEBUG
        os.makedirs(root_dir, exist_ok=True)
        sim_name = get_random_name()
        # END DIR NAMES

        output_dir_path = os.path.join(root_dir, sim_name)
        os.makedirs(output_dir_path, exist_ok=True)

        paths = Paths(output_dir_path)

        print("Results will be stored in ", paths.results_path)
        os.makedirs(os.path.join(paths.results_path, 'models'), exist_ok=True)

        with open(os.path.join(paths.data_path, 'config.yaml'), 'w') as f:
            yaml.dump(config.todict(), f, allow_unicode=True,
                      default_flow_style=False)

        print('Calling main')
        if args.debug:
            from ipdb import launch_ipdb_on_exception

            with launch_ipdb_on_exception():
                main(config, paths)
        else:
            main(config, paths)
        print("Results stored in ", paths.results_path)

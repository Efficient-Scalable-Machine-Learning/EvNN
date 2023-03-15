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
import random
import time
from functools import partialmethod

import numpy as np
import tensorboardX
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from tqdm import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

from models import get_model
from utils import count_parameters, repackage_hidden
from opts import parse_opts, store_args
from utils import AverageMeter, get_dvs128_test_dataset, resume_model, get_dvs128_train_val


def val_epoch(model, data_loader, epoch, criterion, opt, device):
    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()
    mean_activities = AverageMeter()

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(data_loader):
            data, targets = data.to(device), targets.to(device)
            hidden = model.init_hidden(opt.batch_size, device)
            output, hidden, output_gate, _ = model(data.type(torch.float), hidden)
            output_gate_vals = torch.where(output_gate == 0, torch.zeros_like(output_gate),
                                           torch.ones_like(output_gate)).to(
                    device)
            mean_activity = 1.0 - torch.isclose(output_gate.detach().to('cpu'), torch.Tensor([0.0])).to(
                    torch.float).mean()

            if targets.dim() > 1:
                targets = torch.argmax(targets, dim=1)
            loss = criterion(output, targets.type(torch.long))
            preds = output.argmax(dim=1)
            acc = (preds == targets).sum().float() / opt.batch_size

            losses.update(loss.item(), opt.batch_size)
            accuracies.update(acc, opt.batch_size)
            mean_activities.update(mean_activity, opt.batch_size)

    # show info
    print(
            'Validation set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(len(data_loader.dataset),
                                                                                       losses.avg,
                                                                                       accuracies.avg * 100))
    return losses.avg, accuracies.avg, mean_activities.avg


def train_epoch(model, data_loader, criterion, optimizer, epoch, opt, device, tensorboard_writer=None, profile_batch=0):
    log_interval = opt.log_interval
    num_batches = len(data_loader)
    start_time = time.time()
    model.train()

    train_loss = 0.0
    losses = AverageMeter()
    accuracies = AverageMeter()
    mean_activities = AverageMeter()

    for batch_idx, (data, targets) in enumerate(tqdm(data_loader)):
        data, targets = data.to(device), targets.to(device)
        hidden = model.init_hidden(opt.batch_size, device)
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        outputs, hidden, output_gate, _ = model(data.type(torch.float), hidden)

        if targets.dim() > 1:
            targets = torch.argmax(targets, dim=1)
        loss = criterion(outputs, targets.type(torch.long)).to(device)
        acc = (outputs.argmax(dim=1) == targets).sum().float() / opt.batch_size

        output_gate_vals = torch.where(output_gate == 0, torch.zeros_like(output_gate),
                                       torch.ones_like(output_gate)).to(device)
        mean_activity = torch.mean(output_gate_vals).to(device)

        train_loss += loss.item()
        losses.update(loss.item(), opt.batch_size)
        accuracies.update(acc, opt.batch_size)
        mean_activity_detached = 1.0 - torch.isclose(output_gate.detach().to('cpu'), torch.Tensor([0.0])).to(
                torch.float).mean()
        mean_activities.update(mean_activity_detached, opt.batch_size)

        reg_loss = (mean_activity - opt.activity_regularization_target) ** 2

        thr = list(filter(lambda x: 'thr' in x, model.state_dict().keys()))
        hidden = torch.split(hidden, hidden.size(1) // opt.num_layers, dim=1)
        voltage_reg_loss = 0.
        if len(thr) > 0:
            for i, t in enumerate(thr):
                threshold = model.state_dict().get(t).detach()
                voltage_reg_loss += torch.mean(
                        (hidden[i] - (threshold - opt.voltage_regularization_target)))

        if opt.activity_regularization:  # and it > 200:
            total_loss = loss + opt.activity_regularization_constant * reg_loss \
                         + opt.activity_regularization_constant * voltage_reg_loss
        else:
            total_loss = loss

        total_loss.backward()
        optimizer.step()

        if batch_idx == profile_batch:
            for i, t in enumerate(thr):
                threshold = model.state_dict().get(t).detach()
                voltage_minus_thr = (hidden[i].detach() - threshold).to('cpu')
                if tensorboard_writer is not None:
                    tensorboard_writer.add_histogram(f'voltage_dist_layer_{i}',
                                                     voltage_minus_thr.numpy(), epoch)

        if (batch_idx + 1) % log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            avg_loss = train_loss / log_interval
            print(f'| epoch {epoch:3d} | {batch_idx:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.4f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {avg_loss:5.4f} | Acc {accuracies.avg * 100:8.2f} | mean activity {mean_activities.avg:.4f}')
            start_time = time.time()
            train_loss = 0.0

    print('Train set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(
            len(data_loader.dataset), losses.avg, accuracies.avg * 100))

    return losses.avg, accuracies.avg


def main_worker(opt):
    seed = opt.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CUDA for PyTorch
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    # tensorboard
    import shutil
    run_title = opt.run_title + '_'
    tensorboard_dir = os.path.join(opt.logdir, run_title)
    os.makedirs(tensorboard_dir, exist_ok=True)
    print(f'Tensorboard logged to {os.path.abspath(tensorboard_dir)}')
    print('Backing up current file')
    shutil.copy(os.path.abspath(__file__), tensorboard_dir, follow_symlinks=True)
    store_args(os.path.join(tensorboard_dir, 'args.json'), opt)
    summary_writer = tensorboardX.SummaryWriter(log_dir=tensorboard_dir)

    # defining model
    model = get_model(opt, device, optimized=True)
    count_parameters(model)
    # get data loaders
    train_loader, _ = get_dvs128_train_val(opt, split=1, augmentation=opt.augment_data)
    test_loader = get_dvs128_test_dataset(opt)

    # optimizer
    model_params = list(model.parameters())
    if opt.use_rmsprop:
        print('Using RMSprop.')
        optimizer = torch.optim.RMSprop(model_params, lr=opt.learning_rate, weight_decay=0.9)
    else:
        optimizer = torch.optim.Adam(model_params, lr=opt.learning_rate)

    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_epochs, gamma=opt.lr_gamma)
    criterion = nn.CrossEntropyLoss()

    # resume model
    if opt.resume_path:
        start_epoch = resume_model(opt, model, optimizer)
        best_model = model
        best_model_dict = torch.load(os.path.join(opt.resume_path, 'best_model.pth'), map_location=device)
        best_model.load_state_dict(best_model_dict)
        best_model = best_model.to(device)

    else:
        start_epoch = 1

        best_acc = float('-inf')
        best_model = None

    if opt.resume_path:
        best_loss, best_acc, _ = val_epoch(best_model, test_loader, 1, criterion, opt, device)
    # start training
    for epoch in range(start_epoch, opt.train_epochs + 1):
        epoch_start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch, opt, device,
                                            tensorboard_writer=summary_writer)
        val_loss, val_acc, mean_activity = val_epoch(model, test_loader, epoch, criterion, opt, device)

        elapsed = time.time() - epoch_start_time
        lr = optimizer.param_groups[0]['lr']
        # write summary
        summary_writer.add_scalars('Loss', {'train': train_loss,
                                            'val'  : val_loss}, epoch)
        summary_writer.add_scalars('Acc', {'train': train_acc,
                                           'val'  : val_acc}, epoch)

        # saving weights to checkpoint
        if (epoch) % opt.log_interval == 0:
            # scheduler.step(val_loss)
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
            torch.save(state, os.path.join(tensorboard_dir, f'{opt.rnn_type}-Epoch-{epoch}-Acc-{val_acc}.pth'))
            print("Epoch {} model saved!\n".format(epoch))

        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | lr {lr:02.4f} | '
              f'valid loss {val_loss:5.2f} | valid acc {val_acc:5.2f} | mean activity {mean_activity:.4f}')
        print('-' * 89)

        if val_acc > best_acc:
            best_acc = val_acc
            # best_model = copy.deepcopy(model)
            torch.save(model.state_dict(), os.path.join(tensorboard_dir, 'best_model.pth'))

        scheduler.step()

    test_loss, test_acc, test_mean_activity = val_epoch(best_model, test_loader, epoch, criterion, opt, device)

    print('-' * 89)
    print(f'| end of epochs | lr {lr:02.4f} | '
          f'test loss {test_loss:5.2f} | test acc {test_acc:5.2f} | mean activity {test_mean_activity:.4f}')
    print('-' * 89)


if __name__ == "__main__":
    opt = parse_opts()
    print(opt)
    main_worker(opt)

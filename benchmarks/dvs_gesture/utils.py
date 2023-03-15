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

import tonic
import torch
import torchvision
from prettytable import PrettyTable
from tonic import DiskCachedDataset, SlicedDataset
from tonic.slicers import SliceByTime
from torch.utils.data import DataLoader
from torchvision.transforms import RandomPerspective, RandomResizedCrop, RandomRotation


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def resume_model(opt, model, optimizer):
    """ Resume model
    """
    import glob
    model_path = sorted(glob.glob(os.path.join(opt.resume_path, '*-Epoch-*.pth')))[-1]
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Model Restored from Epoch {}".format(checkpoint['epoch']))
    start_epoch = checkpoint['epoch'] + 1

    return start_epoch


class PadTensors:
    def __init__(self, batch_first: bool = False):
        self.batch_first = batch_first

    def __call__(self, batch):
        # batch contains a list of tuples of structure (sequence, target)
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        data = [item[0] for item in sorted_batch]
        data_padded = torch.nn.utils.rnn.pad_sequence(data)
        lengths = torch.LongTensor([len(x) for x in data])
        if self.batch_first:
            data_padded = torch.transpose(data_padded, 0, 1)
        data = torch.nn.utils.rnn.pack_padded_sequence(data_padded, lengths, batch_first=self.batch_first)
        targets = [item[1] for item in sorted_batch]
        return (data, torch.tensor(targets))


class MergeEvents:
    def __init__(self, method: str = 'mean', flatten: bool = True):
        assert method in ['mean', 'diff', 'bool', 'none'], 'Unknown Method'
        self.method = method
        self.flatten = flatten

    def __call__(self, data):
        if self.method == 'mean':
            data = torch.mean(data.type(torch.float), dim=1)
        elif self.method == 'diff':
            data = data[:, 0, ...] - data[:, 1, ...]
        elif self.method == 'bool':
            data = torch.where(data > 1, 1, 0)
        else:
            pass

        if self.flatten:
            return data.reshape((data.size(0), -1))
        else:
            return data


def get_dvs128_train_val(opt, split=0.85, augmentation=False):
    """ Make dataloaders for train and validation sets
    """
    transform, tr_str = get_transforms(opt)

    dataset = tonic.datasets.DVSGesture(save_to=os.path.join(opt.data, 'train'),
                                        train=True,
                                        transform=None,
                                        target_transform=None)

    train_size = int(split * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    min_time_window = 1.7 * 1e6  # 1.7 s
    overlap = 0
    metadata_path = f'_{min_time_window}_{overlap}_{opt.frame_time}_' + tr_str
    slicer_by_time = SliceByTime(time_window=min_time_window, overlap=overlap, include_incomplete=False)
    train_dataset_timesliced = SlicedDataset(train_set, slicer=slicer_by_time, transform=transform,
                                             metadata_path=None)
    val_dataset_timesliced = SlicedDataset(val_set, slicer=slicer_by_time, transform=transform,
                                           metadata_path=None)

    if opt.event_agg_method == 'none' or opt.event_agg_method == 'mean':
        data_max = 19.0  # commented to save time, re calculate if min_time_window changes
        # i=0
        # for data, _ in train_dataset_timesliced:
        #     temp_max = data.max()
        #     data_max = temp_max if temp_max > data_max else data_max
        #     i=i+1
        #
        # for data, _ in val_dataset_timesliced:
        #     temp_max = data.max()
        #     data_max = temp_max if temp_max > data_max else data_max

        print(f'Max train value: {data_max}')
        norm_transform = torchvision.transforms.Lambda(lambda x: x / data_max)
    else:
        norm_transform = None

    if augmentation:
        post_cache_transform = tonic.transforms.Compose([norm_transform, torch.tensor,
                                                         RandomResizedCrop(
                                                                 tonic.datasets.DVSGesture.sensor_size[:-1],
                                                                 scale=(0.6, 1.0),
                                                                 interpolation=torchvision.transforms.InterpolationMode.NEAREST),
                                                         RandomPerspective(),
                                                         RandomRotation(25)
                                                         ])
    else:
        post_cache_transform = norm_transform
    train_cached_dataset = DiskCachedDataset(train_dataset_timesliced, transform=post_cache_transform,
                                             cache_path=os.path.join(opt.cache, 'diskcache_train' + metadata_path))
    val_cached_dataset = DiskCachedDataset(val_dataset_timesliced, transform=post_cache_transform,
                                           cache_path=os.path.join(opt.cache, 'diskcache_val' + metadata_path))

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_dataset = DataLoader(train_cached_dataset, batch_size=opt.batch_size, shuffle=True,
                               collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, **kwargs)
    val_dataset = DataLoader(val_cached_dataset, batch_size=opt.batch_size,
                             collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, **kwargs)

    print(f"Loaded train dataset with {len(train_dataset.dataset)} samples")
    print(f"Loaded test dataset with {len(val_dataset.dataset)} samples")

    return train_dataset, val_dataset


def get_dvs128_test_dataset(opt):
    """ Make dataloaders for test set
    """
    transform, tr_str = get_transforms(opt)

    test_dataset = tonic.datasets.DVSGesture(save_to=os.path.join(opt.data, 'test'),
                                             train=False,
                                             transform=None,
                                             target_transform=None)

    min_time_window = 1.7 * 1e6  # 1.7 s
    overlap = 0  #
    slicer_by_time = SliceByTime(time_window=min_time_window, overlap=overlap, include_incomplete=False)
    # os.makedirs(os.path.join(opt.cache, 'test'), exist_ok=True)
    metadata_path = f'_{min_time_window}_{overlap}_{opt.frame_time}_' + tr_str
    test_dataset_timesliced = SlicedDataset(test_dataset, slicer=slicer_by_time, transform=transform,
                                            metadata_path=None)

    if opt.event_agg_method == 'none' or opt.event_agg_method == 'mean':
        data_max = 18.5  # commented to save time, re calculate if min_time_window changes
        # for data, _ in test_dataset_timesliced:
        #     temp_max = data.max()
        #     data_max = temp_max if temp_max > data_max else data_max

        print(f'Max test value: {data_max}')
        norm_transform = torchvision.transforms.Lambda(lambda x: x / data_max)
    else:
        norm_transform = None

    cached_test_dataset_time = DiskCachedDataset(test_dataset_timesliced, transform=norm_transform,
                                                 cache_path=os.path.join(opt.cache, 'diskcache_test' + metadata_path))
    cached_test_dataloader_time = DataLoader(cached_test_dataset_time, batch_size=opt.batch_size,
                                             collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True)

    print(f"Loaded test dataset with {len(test_dataset)} samples")
    print(f"Loaded sliced test dataset with {len(cached_test_dataset_time)} samples")

    return cached_test_dataloader_time


def get_transforms(opt):
    denoise_transform = tonic.transforms.Denoise(filter_time=10000)
    sensor_size = tonic.datasets.DVSGesture.sensor_size
    frame_transform_time = tonic.transforms.ToFrame(sensor_size=sensor_size,
                                                    time_window=opt.frame_time * 1000,
                                                    include_incomplete=False)

    transform = tonic.transforms.Compose([
        # denoise_transform,
        frame_transform_time,
    ])
    return transform, 'toframe'


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if h is not None:
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters", "Sparsity", "Non-Zero Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        zeros = torch.isclose(parameter, torch.zeros_like(parameter)).sum().cpu().numpy()
        table.add_row([name, param, zeros / param, param - zeros])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params:,}")
    return total_params

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

import evnn_pytorch as evnn
import tonic
import torch
from torch import nn as nn
from torch.nn.utils.rnn import PackedSequence

from dvs_gesture.utils import MergeEvents
from egru import RNNType


class EGRUDDVSModel(nn.Module):
    def __init__(self, input_size, n_units, unit_size, rnn_type, opt):
        super(EGRUDDVSModel, self).__init__()
        self.input_size = input_size[1] * input_size[0]
        self.n_units = n_units
        self.unit_size = unit_size
        self.rnn_type = RNNType(rnn_type)
        self.use_all_timesteps = opt.use_all_timesteps
        print(f'Using RNN: {self.rnn_type.name}')

        self.input_maxpool = nn.MaxPool2d(kernel_size=128 // opt.frame_size)
        # Maxpooling
        self.input_size = self.input_size // ((128 // opt.frame_size) ** 2)
        self.event_merger = MergeEvents(
            method=opt.event_agg_method, flatten=opt.flatten_frame)
        channels = 1 if opt.event_agg_method == 'mean' or opt.event_agg_method == 'diff' else 2

        self.use_cnn = opt.use_cnn
        if self.use_cnn:

            assert not opt.flatten_frame, 'Cannot use cnn after flatten'
            '''
            self.conv = nn.Sequential(nn.Conv2d(in_channels=channels,
                                                out_channels=8,
                                                kernel_size=3,
                                                padding=1),
                                      nn.AvgPool2d(2),
                                      nn.Flatten())
            '''
            self.conv = nn.Sequential(nn.Conv2d(channels, 64, kernel_size=11, stride=4, padding=2),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.Conv2d(
                                          64, 192, kernel_size=5, padding=2),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.Conv2d(
                                          192, 384, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(
                                          kernel_size=3, stride=2) if opt.frame_size >= 64 else nn.Identity(),
                                      nn.Conv2d(
                                          384, 256, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(
                                          kernel_size=3, stride=2) if opt.frame_size >= 128 else nn.Identity(),
                                      nn.Conv2d(
                                          256, 256, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      # nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.Flatten(),
                                      nn.Linear(256, 512)
                                      )
            self.conv = torch.jit.script(self.conv)

            self.input_size = 512
        else:
            self.conv = None
            self.input_size *= channels

        num_layers = opt.num_layers
        self.hidden_dim = self.n_units * self.unit_size
        layer_input_size = [self.input_size] + \
            [self.hidden_dim for _ in range(num_layers - 1)]
        self.layers = []
        for layer_idx in range(num_layers):
            input_size = layer_input_size[layer_idx]
            if self.rnn_type == RNNType.LSTM:
                assert self.unit_size == 1
                rnn = nn.LSTM(input_size, self.hidden_dim, batch_first=True)
            elif self.rnn_type == RNNType.GRU:
                assert self.unit_size == 1
                rnn = nn.GRU(input_size, self.hidden_dim, batch_first=True)
            elif self.rnn_type == RNNType. EGRU:
                rnn = evnn.EGRU(input_size, self.n_units, dropout=opt.dropconnect, zoneout=opt.zoneout,
                                pseudo_derivative_support=opt.pseudo_derivative_width,
                                thr_mean=opt.threshold_mean, batch_first=True)
            else:
                raise RuntimeError("Unknown lstm type: %s" % self.rnn_type)
            print(f"LSTM {layer_idx} parameters: ", list(
                map(lambda x: x[0], rnn.named_parameters())))
            self.layers.append(rnn)
        self.layers = nn.ModuleList(self.layers)

        if opt.dropout:
            self.dropout_layers = nn.ModuleList(
                [nn.Dropout(opt.dropout) for _ in range(num_layers + 1)])
        else:
            self.dropout_layers = None
        self.fc = nn.Linear(self.hidden_dim, len(
            tonic.datasets.DVSGesture.classes))

    def init_hidden(self, batch_size, device):
        return None

    def forward(self, inputs, state):
        seq_len = inputs.size(1)
        batch_size = inputs.size(0)
        lstm_outputs = []
        hidden_states = []
        output_gate_vals = []
        cur_layer_input = inputs
        output_inner = []

        for t in range(seq_len):
            inp = inputs[:, t, ...]
            inp = self.input_maxpool(inp)
            inp = self.event_merger(inp)
            if self.use_cnn:
                inp = self.conv(torch.unsqueeze(inp, 1))
            output_inner.append(inp)

        cur_layer_input = torch.stack(output_inner, dim=1)
        if self.dropout_layers:
            cur_layer_input = self.dropout_layers[0](cur_layer_input)

        for layer_idx, rnn in enumerate(self.layers):
            output_inner = []
            for t in range(seq_len):
                inp = cur_layer_input[:, t, ...]
                output, state = rnn(torch.unsqueeze(inp, 1), state)
                if self.rnn_type == RNNType. EGRU:
                    c_, o_, i_, tr_ = state
                    output_gate_vals.append(o_.squeeze())
                    state = tuple(s[:, -1, ...] for s in state)
                elif self.rnn_type == RNNType.GRU:
                    h_ = c_ = state
                    output_gate_vals.append(h_.squeeze())
                else:
                    h_, c_ = state
                    output_gate_vals.append(h_.squeeze())

                output_inner.append(torch.squeeze(output, dim=1))
                hidden_states.append(torch.squeeze(c_))

            cur_layer_input = torch.stack(output_inner, dim=1)
            if self.dropout_layers:
                cur_layer_input = self.dropout_layers[layer_idx + 1](
                    cur_layer_input)

            lstm_outputs.append(cur_layer_input)

        lstm_out = cur_layer_input
        output_gate = torch.stack(output_gate_vals, dim=1)
        hidden_states = torch.stack(hidden_states).transpose(0, 1)

        if isinstance(inputs, PackedSequence):
            # not implemented for now
            raise NotImplementedError
            # hiddens, lengths = torch.nn.utils.rnn.pad_packed_sequence(hiddens, batch_first=True)

        if not self.use_all_timesteps:
            logits = linear_out = self.fc(
                lstm_out[:, -1, :])  # Last timestep only
        else:
            linear_out = self.fc(lstm_out)
            logits = torch.mean(linear_out, dim=1)  # All time steps

        return logits, hidden_states, output_gate, lstm_outputs


class EGRUDDVSOptimizedModel(EGRUDDVSModel):
    def __init__(self, input_size, n_units, unit_size, rnn_type, opt):
        super(EGRUDDVSOptimizedModel, self).__init__(
            input_size, n_units, unit_size, rnn_type, opt)

    def forward(self, inputs, state):
        seq_len = inputs.size(1)
        batch_size = inputs.size(0)
        outputs = []
        hidden_states = []
        output_gate_vals = []
        cur_layer_input = inputs
        output_inner = []

        inp = inputs.contiguous().view(
            (batch_size * seq_len,) + inputs.size()[2:])
        inp = self.input_maxpool(inp)
        inp = self.event_merger(inp)
        if self.use_cnn:
            inp = self.conv(torch.unsqueeze(inp, 1))

        cur_layer_input = inp.contiguous().view(
            (batch_size, seq_len) + inp.size()[1:])
        if self.dropout_layers:
            cur_layer_input = self.dropout_layers[0](cur_layer_input)

        for layer_idx, rnn in enumerate(self.layers):
            output, state = rnn(cur_layer_input, None)
            if self.rnn_type == RNNType. EGRU:
                c_, o_, tr_ = state
                output = tr_
                output_gate_vals.append(o_.unsqueeze(0))
                c_ = c_.unsqueeze(0)
            elif self.rnn_type == RNNType.GRU:
                h_ = c_ = state
                c_ = c_.unsqueeze(0)
                output_gate_vals.append(output.unsqueeze(0).transpose(1, 2))
            else:
                h_, c_ = state
                c_ = c_.unsqueeze(0)
                output_gate_vals.append(output.unsqueeze(0).transpose(1, 2))

            hidden_states.append(c_)

            cur_layer_input = output
            if self.dropout_layers:
                cur_layer_input = self.dropout_layers[layer_idx + 1](
                    cur_layer_input)

        lstm_out = cur_layer_input
        output_gate = torch.concat(output_gate_vals).permute(
            2, 0, 1, 3)  # (Batch, layer, time, hidden)
        hidden_states = torch.concat(hidden_states).permute(2, 0, 1, 3)

        if isinstance(inputs, PackedSequence):
            # not implemented for now
            raise NotImplementedError
            # hiddens, lengths = torch.nn.utils.rnn.pad_packed_sequence(hiddens, batch_first=True)

        if not self.use_all_timesteps:
            logits = linear_out = self.fc(
                lstm_out[:, -1, :])  # Last timestep only
        else:
            linear_out = self.fc(lstm_out)
            logits = torch.mean(linear_out, dim=1)  # All time steps

        return logits, hidden_states, output_gate, linear_out


def get_model(opt, device, optimized=True):
    if optimized:
        model = EGRUDDVSOptimizedModel(input_size=tonic.datasets.DVSGesture.sensor_size, n_units=opt.units,
                                       unit_size=opt.unit_size, rnn_type=opt.rnn_type, opt=opt)
    else:
        model = EGRUDDVSModel(input_size=tonic.datasets.DVSGesture.sensor_size, n_units=opt.units,
                              unit_size=opt.unit_size, rnn_type=opt.rnn_type, opt=opt)

    print(model)
    return model.to(device)

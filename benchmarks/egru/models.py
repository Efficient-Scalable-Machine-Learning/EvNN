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

from enum import IntEnum

import torch
import torch.nn as nn

torch.autograd.set_detect_anomaly(True)


class RNNWrapper(nn.Module):
    """
    """

    def __init__(self, rnn):
        super(RNNWrapper, self).__init__()
        self.rnn = rnn
        self.hidden_size = rnn.hidden_size

    def forward(self, all_inputs):
        rnn_out, *hidden = self.rnn(all_inputs)

        return rnn_out, hidden


class RNNReadoutWrapper(nn.Module):
    """
    This class puts a readout on top of the passed in RNN.
    For pytorch LSTM, the outputs contain the hidden states of only the last layer.
    BUT DOESN'T USE READOUT. HAS TO BE DONE OUTSIDE.
    """

    def __init__(self, rnn, output_size: int):
        super(RNNReadoutWrapper, self).__init__()
        self.rnn = rnn
        self.output_size = output_size

        # NOTE: That there is no sigmoid here, since it's applied at the loss function
        self.hidden2out = nn.Linear(self.rnn.hidden_size, self.output_size)

    def forward(self, all_inputs):
        rnn_out, hidden = self.rnn(all_inputs)

        return rnn_out, *hidden


from egru.reg_modules import VariationalDropout


class RNNMultiLayerWrapper(nn.Module):
    """
    """

    def __init__(self, rnns, dropout_forward=0.):
        super(RNNMultiLayerWrapper, self).__init__()
        self.rnns = nn.ModuleList(rnns)
        self.n_rnns = len(rnns)

        self.hidden_size = rnns[-1].hidden_size
        self.dropout_forward = dropout_forward
        self.variational_dropout = VariationalDropout()

    def forward(self, ext_inputs):
        inputs = ext_inputs
        if self.n_rnns == 1:
            rnn_out, states = self.rnns[0](inputs)
        else:
            for model in self.rnns:
                rnn_out, states = model(inputs)
                rnn_out = self.variational_dropout(rnn_out, dropout=self.dropout_forward)
                inputs = rnn_out

        return rnn_out, states

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

import torch
import torch.nn as nn
import torch.nn.functional as F
import evnn_pytorch as evnn
from lm.modules import VariationalDropout, WeightDrop
from lm.embedding_dropout import embedded_dropout
from typing import Union


class Decoder(nn.Module):
    def __init__(self,
                 ninp: int,
                 ntokens: int,
                 project: bool = False,
                 nemb: Union[None, int] = None,
                 dropout: float = 0.0):
        """
        Takes hidden states of RNNs, optionally applies a projection operation and decodes to output tokens
        :param ninp: Input dimension
        :param ntokens: Number of tokens of the language model
        :param project: If True, applies a linear projection onto the embedding dimension
        :param nemb: If projection is True, specifies the dimension of the projection
        :param dropout: Dropout rate applied to the projector
        """
        super(Decoder, self).__init__()

        if project:
            assert nemb, "If projection is True, must specify nemb!"

        self.ninp = ninp
        self.nemb = nemb if nemb else ninp
        self.nout = ntokens

        self.dropout = dropout
        self.variational_dropout = VariationalDropout()

        # projector
        self.project = project
        if project:
            self.projection = nn.Linear(ninp, nemb)
        else:
            self.projection = nn.Identity()

        # word embedding decoder
        self.decoder = nn.Linear(self.nemb, self.nout)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        bs, seq_len, ninp = x.shape
        if self.project:
            x = x.view(-1, ninp)
            x = F.relu(self.projection(x))
            x = x.view(bs, seq_len, self.nemb)
            x = self.variational_dropout(x, self.dropout)
        x = x.view(-1, self.nemb)
        x = self.decoder(x)
        return x


class LanguageModel(nn.Module):
    def __init__(self,
                 rnn_type,
                 nlayers,
                 emb_dim,
                 hidden_dim,
                 vocab_size,
                 projection,
                 dropout_words,
                 dropout_embedding,
                 dropout_connect,
                 dropout_forward,
                 alpha,
                 beta,
                 gamma,
                 **kwargs):
        super(LanguageModel, self).__init__()

        # language model specifics
        self.nlayers = nlayers
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.projection = projection

        # dropout initializations
        self.dropout_words = dropout_words
        self.dropout_embedding = dropout_embedding
        self.dropout_connect = dropout_connect
        self.dropout_forward = dropout_forward

        # activity regularization specification
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ar_loss = 0

        # input and output layers
        self.variational_dropout = VariationalDropout()
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        initrange = 0.1
        self.init_embedding(initrange=initrange)

        self.decoder = Decoder(ninp=hidden_dim if projection else emb_dim, ntokens=vocab_size,
                               project=projection, nemb=emb_dim,
                               dropout=dropout_forward)

        # Tie weights of embedding and decoder
        self.decoder.decoder.weight = self.embeddings.weight

        # RNN model definition
        self.rnn_type = rnn_type
        if rnn_type == 'lstm':
            self.rnns = [nn.LSTM(emb_dim if l == 0 else hidden_dim,
                                 emb_dim if l == nlayers - 1 and not projection else hidden_dim,
                                 num_layers=1, batch_first=True, dropout=0)
                         for l in range(nlayers)]
            if dropout_connect > 0:
                self.rnns = [WeightDrop(
                    rnn, [f'weight_hh_l0'], dropout=dropout_connect) for rnn in self.rnns]
        elif rnn_type == 'gru':
            self.rnns = [nn.GRU(emb_dim if l == 0 else hidden_dim,
                                emb_dim if l == nlayers - 1 and not projection else hidden_dim,
                                num_layers=1, batch_first=True, dropout=0)
                         for l in range(nlayers)]
            if dropout_connect > 0:
                self.rnns = [WeightDrop(
                    rnn, ['weight_hh_l0'], dropout=dropout_connect) for rnn in self.rnns]
        elif rnn_type == 'egru':
            self.rnns = [evnn.EGRU(input_size=emb_dim if l == 0 else hidden_dim,
                                   hidden_size=emb_dim if l == nlayers - 1 and not projection else hidden_dim,
                                   batch_first=True,
                                   dropout=dropout_connect,
                                   zoneout=0.0,
                                   return_state_sequence=False,
                                   use_custom_cuda=True,
                                   **kwargs) for l in range(nlayers)]
        else:
            raise NotImplementedError(f"Model '{rnn_type}' not implemented.")
        self.rnns = nn.ModuleList(self.rnns)

        self.backward_sparsity = torch.zeros(len(self.rnns))

    def prune_embeddings(self, index):
        device = next(self.parameters()).device
        self.embeddings.weight = nn.Parameter(
            self.embeddings.weight[:, index]).to(device)
        self.emb_dim = self.embeddings.weight.shape[1]
        self.decoder = Decoder(ninp=self.hidden_dim if self.projection else self.emb_dim, ntokens=self.vocab_size,
                               project=self.projection, nemb=self.emb_dim,
                               dropout=self.dropout_forward).to(device)
        self.decoder.decoder.weight = self.embeddings.weight

    def prune(self, fractions, hiddens, device):
        # calculate new hidden dimensions
        indicies = [torch.arange(self.rnns[0].input_size).to(device)]

        for i in range(self.nlayers):
            if isinstance(fractions, float):
                frac = fractions
            elif isinstance(fractions, tuple) or isinstance(fractions, list):
                frac = fractions[i]
            else:
                raise NotImplementedError(
                    f"data type {type(fractions)} not implemented. Use float, tuple or list")

            # get event frequencies
            hid_dim = hiddens[i].shape[2]
            hid_cells = hiddens[i].reshape(-1, hid_dim)
            seq_len = hid_cells.shape[0]
            spike_frequency = torch.sum(hid_cells != 0, dim=0) / seq_len
            print(
                f"Layer {i + 1}: "
                f"less than 1/100: {torch.sum(spike_frequency < 0.01)} / {spike_frequency.shape} "
                f"// never: {torch.sum(hid_cells.sum(dim=0) == 0)} / {spike_frequency.shape}")

            # compute remaining indicies from spike frequencies
            topk = int(self.rnns[i].hidden_size * (1 - frac))
            hidden_indices, _ = torch.sort(torch.argsort(
                spike_frequency, descending=True)[:topk], descending=False)
            hidden_indices = hidden_indices.to(device)
            indicies.append(hidden_indices)

        # input dimension equals embedding dimension for tied weights
        indicies[0] = indicies[-1]

        # prune weights
        for i in range(self.nlayers):
            self.rnns[i].prune_units(indicies[i], indicies[i+1])

        self.prune_embeddings(indicies[-1])
        print(
            f"Final model hidden size: {[rnn.hidden_size for rnn in self.rnns]}")

    def init_embedding(self, initrange):
        nn.init.uniform_(self.embeddings.weight, -initrange, initrange)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == 'lstm':
            return [(weight.new_zeros(1, batch_size, self.emb_dim if l == self.nlayers - 1 and not self.projection else self.hidden_dim),
                     weight.new_zeros(1, batch_size, self.emb_dim if l == self.nlayers - 1 and not self.projection else self.hidden_dim))
                    for l in range(self.nlayers)]

        elif self.rnn_type == 'gru':
            return [weight.new_zeros(1, batch_size, self.emb_dim if l == self.nlayers - 1 and not self.projection else self.hidden_dim)
                    for l in range(self.nlayers)]

        elif self.rnn_type == 'egru':
            return [None for _ in self.rnns]

        else:
            raise NotImplementedError(
                f"Model '{self.rnn_type}' not implemented.")

    def forward(self, inputs, state):
        # reset activity regularization loss
        self.ar_loss = 0

        # embedding forward
        embedded = embedded_dropout(self.embeddings, inputs,
                                    dropout=self.dropout_words if self.training else 0)

        embedded = self.variational_dropout(
            embedded, dropout=self.dropout_embedding)

        # rnn forward
        new_states = []
        raw_hiddens = []
        dropped_hiddens = []
        hiddens = embedded
        for l, rnn in enumerate(self.rnns):
            hiddens, final_states = rnn(hiddens, state[l])

            raw_hiddens.append(hiddens)

            if self.rnn_type == 'egru':
                c, o, tr = final_states

                self.ar_loss += self.activity_regularization(
                    hidden_raw=c, output_gates=o)
                self.set_backward_sparsity(l, c)

                # Network outputs full states, but takes only the last state as input
                final_states = torch.unsqueeze(hiddens[:, -1], dim=0)

            new_states.append(final_states)

            if l != self.nlayers - 1:
                hiddens = self.variational_dropout(
                    hiddens, dropout=self.dropout_forward)
                dropped_hiddens.append(hiddens)

            if l == self.nlayers - 1 and self.rnn_type != 'egru':
                self.ar_loss += self.activity_regularization(hiddens)

        # decoder forward
        hiddens_ = hiddens.contiguous()
        hiddens_ = self.variational_dropout(hiddens_)
        dropped_hiddens.append(hiddens_)

        decoded = self.decoder(hiddens_)
        return F.log_softmax(decoded, dim=1), new_states, raw_hiddens, dropped_hiddens

    def set_backward_sparsity(self, l, cell_states):

        thresholds = self.rnns[l].module.thr if isinstance(
            self.rnns[l], WeightDrop) else self.rnns[l].thr
        thresholds = thresholds.to(cell_states.device)

        if self.rnn_type == 'egru':
            epsilon = self.rnns[l].pseudo_derivative_support
        else:
            raise NotImplementedError(
                f"RNN type {self.rnn_type} does not implement surrogate gradient")

        self.backward_sparsity[l] = torch.mean(
            torch.logical_or((cell_states - thresholds) > 1 / epsilon,
                             (cell_states - thresholds) < - 1 / epsilon).float())

    def activity_regularization(self, hidden_raw, hidden_dropped=None, output_gates=None):
        ar, tar, gates = 0, 0, 0

        hidden_raw = hidden_raw.squeeze()
        hidden_dropped = hidden_raw.squeeze() if torch.is_tensor(
            hidden_dropped) else hidden_raw
        output_gates = output_gates.squeeze() if torch.is_tensor(
            output_gates) else output_gates

        # EGRU
        if self.rnn_type == 'egru':

            # regularize activity to approach its minimum (-1)
            if self.alpha:
                state_reg = torch.mean(hidden_raw)
                ar = self.alpha * state_reg

            # regularize differences in states
            if self.beta:
                state_diff = hidden_raw[1:] - hidden_raw[:-1]
                mask = output_gates[:-1].bool()
                temporal_state_reg = torch.mean(state_diff[mask].pow(2))
                tar = self.beta * temporal_state_reg

            # regularize activity
            if self.gamma:
                activity_reg = torch.mean(output_gates)
                gates = self.gamma * activity_reg

        # LSTM, GRU etc
        else:
            hidden_dropped = hidden_dropped if torch.is_tensor(
                hidden_dropped) else hidden_raw
            if self.alpha:
                ar = self.alpha * hidden_dropped.pow(2).mean()
            if self.beta:
                tar = self.beta * \
                    (hidden_raw[:, 1:] - hidden_raw[:, :-1]).pow(2).mean()

        return ar + tar + gates

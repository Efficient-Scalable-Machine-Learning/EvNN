# Copyright (c) 2023  Khaleelulla Khan Nazeer
# This file incorporates work covered by the following copyright:
# Copyright 2020 LMNT, Inc. All Rights Reserved.
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

"""Gated Recurrent Unit"""
import math

import evnn_pytorch_lib as LIB
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_rnn import BaseRNN

__all__ = [
    'EGRU'
]


class SpikeFunction(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, inp, dampening_factor, pseudo_derivative_support):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(inp, dampening_factor, pseudo_derivative_support)
        return torch.heaviside(inp, inp)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        inp, dampening_factor, pseudo_derivative_support = ctx.saved_tensors
        dE_dz = grad_output

        dz_du = dampening_factor * torch.maximum(1 - pseudo_derivative_support * torch.abs(
            inp), torch.Tensor((0,)).to(grad_output.device))

        dE_dv = dE_dz * dz_du
        return dE_dv, None, None


# @torch.jit.script
def EGRUScript(
        training: bool,
        zoneout_prob: float,
        dampening_factor: float,
        pseudo_derivative_support: float,
        input,
        h0,
        kernel,
        recurrent_kernel,
        bias,
        recurrent_bias,
        thr,
        zoneout_mask):
    """
    Perform EGRU computation using Pytorch primitives.

    :param training: bool,
    :type training: bool
    :param zoneout_prob: the probability of zoneout
    :type zoneout_prob: float
    :param dampening_factor: This is the dampening factor for the spike function
    :type dampening_factor: float
    :param pseudo_derivative_support: float,
    :type pseudo_derivative_support: float
    :param input: the input to the RNN, of shape (time_steps, batch_size, input_size)
    :param h0: initial hidden state
    :param kernel: the input weight matrix
    :param recurrent_kernel: the recurrent weight matrix
    :param bias: bias vector
    :param recurrent_bias: bias for recurrent kernel
    :param thr: threshold
    :param zoneout_mask: a mask that is used to randomly set some of the hidden units to zero
    :return: The output of the EGRU cell, the hidden state, the output of the spike function, and the
    trace values.
    """

    time_steps = input.shape[0]
    batch_size = input.shape[1]
    hidden_size = recurrent_kernel.shape[0]

    h = [torch.zeros_like(h0)]
    o = [torch.zeros_like(h0)]
    y = [h0]
    Wx = input @ kernel + bias
    for t in range(time_steps):
        Rh = y[t] @ recurrent_kernel + recurrent_bias
        vx = torch.chunk(Wx[t], 3, 1)
        vh = torch.chunk(Rh, 3, 1)

        z = torch.sigmoid(vx[0] + vh[0])
        r = torch.sigmoid(vx[1] + vh[1])
        g = torch.tanh(vx[2] + r * vh[2])

        cur_h = (z * h[t] + (1 - z) * g)
        if zoneout_prob:
            if training:
                cur_h = (cur_h - h[t]) * zoneout_mask[t] + h[t]
            else:
                cur_h = zoneout_prob * h[t] + (1 - zoneout_prob) * h[t]

        event = SpikeFunction.apply(
            cur_h - thr, dampening_factor, pseudo_derivative_support)
        o.append(event)
        h.append(cur_h - event * thr)
        y.append(event * cur_h)

    y = torch.stack(y)
    h = torch.stack(h)
    o = torch.stack(o)

    tr_vals = torch.zeros_like(y)
    alpha = 0.9
    for t in range(1, time_steps + 1):
        tr_vals[t] = alpha * tr_vals[t - 1] + (1 - alpha) * y[t]

    return y, h, o, tr_vals


class EGRUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, training, zoneout_prob, dampening_factor, pseudo_derivative_support, max_norm, *inputs):
        '''
        Function call signature from EGRU.forward 
          self.training,
          self.zoneout,
          self.dampening_factor,
          self.pseudo_derivative_support,
          self._max_norm,
          inputs -,
                  | input.contiguous(),
                  | state.contiguous(),
                  | self.kernel.contiguous(),
                  | F.dropout(self.recurrent_kernel, self.dropout, self.training).contiguous(),
                  | self.bias.contiguous(),
                  | self.recurrent_bias.contiguous(),
                  | thr.contiguous(),
                  | zoneout_mask.contiguous()
        '''
        if inputs[0].is_cuda and 'egru_forward' in LIB.__dict__:
            egru_forward = LIB.egru_forward
        else:
            egru_forward = LIB.egru_forward_cpu

        y, cache, h, o, trace = egru_forward(training, zoneout_prob, *inputs)
        ctx.save_for_backward(inputs[0], *inputs[2:], dampening_factor,
                              pseudo_derivative_support, max_norm, y, h, cache)
        ctx.mark_non_differentiable(inputs[-1])
        ctx.training = training
        return y, h, o, trace

    @staticmethod
    def backward(ctx, grad_y, grad_h, grad_o, grad_trace):

        # uncomment to enable breakpoint
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        if not ctx.training:
            raise RuntimeError(
                'EGRU backward can only be called in training mode')

        saved = [*ctx.saved_tensors]
        saved[0] = saved[0].permute(2, 0, 1).contiguous()
        saved[1] = saved[1].permute(1, 0).contiguous()
        saved[2] = saved[2].permute(1, 0).contiguous()
        '''
    saved-, 
          | input.contiguous(),
          | self.kernel.contiguous(),
          | F.dropout(self.recurrent_kernel, self.dropout, self.training).contiguous(),
          | self.bias.contiguous(),
          | self.recurrent_bias.contiguous(),
          | thr.contiguous(),
          | zoneout_mask.contiguous(),
          | dampening_factor,
          | pseudo_derivative_support,
          | max_norm,
          | y,
          | h,
          | cache

    '''
        # for t in range(grad_trace.size(0)-1,0, -1):
        #   grad_trace[t-1] += 0.9 * grad_trace[t]
        # grad_y += 0.1 * grad_trace

        if saved[0].is_cuda and 'egru_backward' in LIB.__dict__:
            egru_backward = LIB.egru_backward
        else:
            egru_backward = LIB.egru_backward_cpu
        *grads, grad_scaling_factor = egru_backward(*saved, grad_y.contiguous(
        ), grad_h.contiguous(), grad_o.contiguous(), grad_trace.contiguous())
        grads = grads[:-2]
        if grad_scaling_factor < 1e-06:
            print('grads scaled by {}'.format(grad_scaling_factor.item()))
        '''
    grads-,
          | dx,
          | dy, 
          | dW, 
          | dR, 
          | dbx, 
          | dbr, 
          | dthr
    '''
        return (None, None, None, None, None, *grads, None)


class EGRU(BaseRNN):
    """
    Event based Gated Recurrent Unit layer.

    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 batch_first=False,
                 dropout=0.0,
                 zoneout=0.0,
                 dampening_factor=0.7,
                 pseudo_derivative_support=1.0,
                 thr_mean=0.3,
                 weight_initialization_gain=1.0,
                 return_state_sequence=False,
                 grad_clip=None,
                 use_custom_cuda=True):
        """
        Initialize the parameters of the GRU layer.

        Arguments:
          input_size: int, the feature dimension of the input.
          hidden_size: int, the feature dimension of the output.
          batch_first: (optional) bool, if `True`, then the input and output
            tensors are provided as `(batch, seq, feature)`.
          dropout: (optional) float, sets the dropout rate for DropConnect
            regularization on the recurrent matrix.
          zoneout: (optional) float, sets the zoneout rate for Zoneout
            regularization.
          return_state_sequence: (optional) bool, if `True`, the forward pass will
            return the entire state sequence instead of just the final state. Note
            that if the input is a padded sequence, the returned state will also
            be a padded sequence.
          grad_clip: (optional) float, sets the gradient clipping value.
          use_custom_cuda (optional) bool, if `True`, the cuda code is used else 
            pytorch implementation is used.

        """
        super().__init__(input_size, hidden_size, batch_first, zoneout, return_state_sequence)
        self.use_custom_cuda = use_custom_cuda

        if grad_clip:
            self.grad_clip_norm(enable=True, norm=grad_clip)
        else:
            self.grad_clip_norm(False)

        if dropout < 0 or dropout > 1:
            raise ValueError('GRU: dropout must be in [0.0, 1.0]')
        if zoneout < 0 or zoneout > 1:
            raise ValueError('GRU: zoneout must be in [0.0, 1.0]')

        self.dropout = dropout
        self.alpha = torch.tensor(0.9)

        self.weight_initialization_gain = weight_initialization_gain

        self.kernel = nn.Parameter(torch.empty(input_size, hidden_size * 3))
        self.recurrent_kernel = nn.Parameter(
            torch.empty(hidden_size, hidden_size * 3))
        self.bias = nn.Parameter(torch.empty(hidden_size * 3))
        self.recurrent_bias = nn.Parameter(torch.empty(hidden_size * 3))
        self.reset_parameters()

        self.dampening_factor = nn.Parameter(
            torch.Tensor([dampening_factor]), requires_grad=False)
        self.pseudo_derivative_support = nn.Parameter(
            torch.Tensor([pseudo_derivative_support]), requires_grad=False)

        # initialize thresholds according to the beta distribution with mean 'thr_mean'
        assert 0 < thr_mean < 1, f"thr_mean must be between 0 and 1, but {thr_mean} was given"
        beta = 3
        alpha = beta * thr_mean / (1 - thr_mean)
        distribution = torch.distributions.beta.Beta(alpha, beta)
        self.thr = nn.Parameter(distribution.sample(torch.Size([self.hidden_size])))

    def to_native_weights(self):
        """
        Converts EvNN GRU weights to native PyTorch GRU weights.

        Returns:
          weight_ih_l0: Parameter, the input-hidden weights of the GRU layer.
          weight_hh_l0: Parameter, the hidden-hidden weights of the GRU layer.
          bias_ih_l0: Parameter, the input-hidden bias of the GRU layer.
          bias_hh_l0: Parameter, the hidden-hidden bias of the GRU layer.
        """
        def reorder_weights(w):
            z, r, n = torch.chunk(w, 3, dim=-1)
            return torch.cat([z, r, n], dim=-1)

        kernel = reorder_weights(self.kernel).permute(1, 0).contiguous()
        recurrent_kernel = reorder_weights(
            self.recurrent_kernel).permute(1, 0).contiguous()
        bias1 = reorder_weights(self.bias).contiguous()
        bias2 = reorder_weights(self.recurrent_bias).contiguous()

        kernel = torch.nn.Parameter(kernel)
        recurrent_kernel = torch.nn.Parameter(recurrent_kernel)
        bias1 = torch.nn.Parameter(bias1)
        bias2 = torch.nn.Parameter(bias2)
        thr = torch.nn.Parameter(self.thr)
        return kernel, recurrent_kernel, bias1, bias2, thr

    def from_native_weights(self, weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0, thr):
        """
        Copies and converts the provided PyTorch GRU weights into this layer.

        Arguments:
          weight_ih_l0: Parameter, the input-hidden weights of the PyTorch GRU layer.
          weight_hh_l0: Parameter, the hidden-hidden weights of the PyTorch GRU layer.
          bias_ih_l0: Parameter, the input-hidden bias of the PyTorch GRU layer.
          bias_hh_l0: Parameter, the hidden-hidden bias of the PyTorch GRU layer.
        """
        def reorder_weights(w):
            z, r, n = torch.chunk(w, 3, axis=-1)
            return torch.cat([z, r, n], dim=-1)

        kernel = reorder_weights(weight_ih_l0.permute(1, 0)).contiguous()
        recurrent_kernel = reorder_weights(
            weight_hh_l0.permute(1, 0)).contiguous()
        bias = reorder_weights(bias_ih_l0).contiguous()
        recurrent_bias = reorder_weights(bias_hh_l0).contiguous()

        self.kernel = nn.Parameter(kernel)
        self.recurrent_kernel = nn.Parameter(recurrent_kernel)
        self.bias = nn.Parameter(bias)
        self.recurrent_bias = nn.Parameter(recurrent_bias)
        self.thr = nn.Parameter(thr)

    def reset_parameters(self):
        """Resets this layer's parameters to their initial values."""
        for k, v in self.named_parameters():
            if k in ['kernel', 'recurrent_kernel', 'bias', 'recurrent_bias']:
                if v.data.ndimension() >= 2:
                    nn.init.xavier_normal_(v, gain=self.weight_initialization_gain)
                else:
                    nn.init.zeros_(v)

    def grad_clip_norm(self, enable=True, norm=2.0):
        self._enable_grad_clip = enable
        self._max_norm = nn.Parameter(torch.Tensor(
            [norm if enable else -1.0]), requires_grad=False)

    def forward(self, input, state=None, lengths=None):
        """
        Runs a forward pass of the EGRU layer.

        Arguments:
          input: Tensor, a batch of input sequences to pass through the GRU.
            Dimensions (seq_len, batch_size, input_size) if `batch_first` is
            `False`, otherwise (batch_size, seq_len, input_size).
          lengths: (optional) Tensor, list of sequence lengths for each batch
            element. Dimension (batch_size). This argument may be omitted if
            all batch elements are unpadded and have the same sequence length.

        Returns:
          output: Tensor, the output of the EGRU layer. Dimensions
            (seq_len, batch_size, hidden_size) if `batch_first` is `False` (default)
            or (batch_size, seq_len, hidden_size) if `batch_first` is `True`. Note
            that if `lengths` was specified, the `output` tensor will not be
            masked. It's the caller's responsibility to either not use the invalid
            entries or to mask them out before using them.
          h: the hidden state for all sequences. Dimensions
            (seq_len, batch_size, hidden_size).
          o: the output gate for all sequences (values: 0 or 1).
          trace: smoothed output values, can be beneficial for training.
        """
        input = self._permute(input)
        state_shape = [1, input.shape[1], self.hidden_size]
        h0 = self._get_state(input, state, state_shape)

        # restrict thresholds to be between 0 and 1
        self.thr.data.clamp_(min=0.0, max=1.0)

        # run forward pass
        y, h, o, trace = self._impl(
            input, h0[0], self.thr, self._get_zoneout_mask(input))

        # prepare outputs
        output = self._permute(y[1:])
        h = self._permute(h[1:])
        o = self._permute(o[1:])
        trace = self._permute(trace[1:])
        return output, (h, o, trace)

    def _impl(self, input, state, thr, zoneout_mask):
        if self.use_custom_cuda:
            return EGRUFunction.apply(
                self.training,
                self.zoneout,
                self.dampening_factor,
                self.pseudo_derivative_support,
                self._max_norm,
                input.contiguous(),
                state.contiguous(),
                self.kernel.contiguous(),
                F.dropout(self.recurrent_kernel, self.dropout,
                          self.training).contiguous(),
                self.bias.contiguous(),
                self.recurrent_bias.contiguous(),
                thr.contiguous(),
                zoneout_mask.contiguous())
        else:
            return EGRUScript(
                self.training,
                self.zoneout,
                self.dampening_factor,
                self.pseudo_derivative_support,
                input.contiguous(),
                state.contiguous(),
                self.kernel.contiguous(),
                F.dropout(self.recurrent_kernel, self.dropout,
                          self.training).contiguous(),
                self.bias.contiguous(),
                self.recurrent_bias.contiguous(),
                thr.contiguous(),
                zoneout_mask.contiguous())

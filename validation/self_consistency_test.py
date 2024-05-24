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

import unittest
from unittest import mock
import torch
import evnn_pytorch as evnn

import numpy as np


RNN_MAP = {
    'egru': evnn.EGRU,
}


batch_size = 10
time_steps = 8
input_size = 4
hidden_size = 8
error_tol = 1e-3

@unittest.skipUnless(torch.cuda.is_available(), 'CUDA not available')
class EGRUCUDAForwardTest(unittest.TestCase):
    def setUp(self):
        rnn = RNN_MAP['egru']
        self.seed = 5526

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)

        x = torch.rand(batch_size, time_steps, input_size, dtype=torch.float32)
        self.egru = rnn(input_size, hidden_size, zoneout=0.0, batch_first=True)
        self.egru.cuda()

        self.x_cuda = x.clone().cuda()
        self.x_cuda_torch = self.x_cuda.detach().clone()

    def test_forward_y(self):
        with torch.no_grad():
            y1, _ = self.egru.forward(self.x_cuda)

            torch.manual_seed(self.seed)
            with mock.patch.object(self.egru, "use_custom_cuda", False):
                y2, _ = self.egru.forward(self.x_cuda_torch)

            assert torch.allclose(y1, y2, atol=error_tol)

    def test_forward_h(self):
        with torch.no_grad():
            _, (h1, _, _) = self.egru.forward(self.x_cuda)

            torch.manual_seed(self.seed)
            with mock.patch.object(self.egru, "use_custom_cuda", False):
                _, (h2, _, _) = self.egru.forward(self.x_cuda_torch)

            assert torch.allclose(h1, h2, atol=error_tol)

    def test_forward_o(self):
        with torch.no_grad():
            _, (_, o1, _) = self.egru.forward(self.x_cuda)

            torch.manual_seed(self.seed)
            with mock.patch.object(self.egru, "use_custom_cuda", False):
                _, (_, o2, _) = self.egru.forward(self.x_cuda_torch)

            assert torch.allclose(o1, o2)

    def test_forward_trace(self):
        with torch.no_grad():
            _, (_, _, t1) = self.egru.forward(self.x_cuda)

            torch.manual_seed(self.seed)
            with mock.patch.object(self.egru, "use_custom_cuda", False):
                _, (_, _, t2) = self.egru.forward(self.x_cuda_torch)

            assert torch.allclose(t1, t2, atol=error_tol)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA not available')
class EGRUCUDABackwardTest(unittest.TestCase):
    def setUp(self):
        rnn = RNN_MAP['egru']
        self.seed = 5526

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)

        x = torch.rand(batch_size, time_steps, input_size, dtype=torch.float32)
        self.egru = rnn(input_size, hidden_size, zoneout=0.0, batch_first=True)
        self.egru.cuda()

        self.x_cuda = x.clone().cuda()
        self.x_cuda_torch = self.x_cuda.detach().clone()
        self.x_cuda.requires_grad_(True)
        self.x_cuda_torch.requires_grad_(True)

    def test_backward_y(self):
        y1, _ = self.egru.forward(self.x_cuda)

        torch.manual_seed(self.seed)
        with mock.patch.object(self.egru, "use_custom_cuda", False):
            y2, _ = self.egru.forward(self.x_cuda_torch)

        y1.backward(torch.ones_like(y1), retain_graph=True)
        y2.backward(torch.ones_like(y2), retain_graph=True)

        assert torch.allclose(self.x_cuda_torch.grad.data,
                              self.x_cuda.grad.data, atol=error_tol)

    def test_backward_h(self):
        _, (h1, _, _) = self.egru.forward(self.x_cuda)

        torch.manual_seed(self.seed)
        with mock.patch.object(self.egru, "use_custom_cuda", False):
            _, (h2, _, _) = self.egru.forward(self.x_cuda_torch)

        h1.backward(torch.ones_like(h1), retain_graph=True)
        h2.backward(torch.ones_like(h2), retain_graph=True)

        assert torch.allclose(self.x_cuda_torch.grad.data,
                              self.x_cuda.grad.data, atol=error_tol)

    def test_backward_o(self):
        _, (_, o1, _) = self.egru.forward(self.x_cuda)

        torch.manual_seed(self.seed)
        with mock.patch.object(self.egru, "use_custom_cuda", False):
            _, (_, o2, _) = self.egru.forward(self.x_cuda_torch)

        o1.backward(torch.ones_like(o1), retain_graph=True)
        o2.backward(torch.ones_like(o2), retain_graph=True)

        assert torch.allclose(self.x_cuda_torch.grad.data,
                              self.x_cuda.grad.data, atol=error_tol)

    def test_backward_trace(self):
        _, (_, _, t1) = self.egru.forward(self.x_cuda)

        torch.manual_seed(self.seed)
        with mock.patch.object(self.egru, "use_custom_cuda", False):
            _, (_, _, t2) = self.egru.forward(self.x_cuda_torch)

        t1.backward(torch.ones_like(t1), retain_graph=True)
        t2.backward(torch.ones_like(t2), retain_graph=True)

        assert torch.allclose(self.x_cuda_torch.grad.data,
                              self.x_cuda.grad.data, atol=error_tol)


if __name__ == '__main__':
    unittest.main()

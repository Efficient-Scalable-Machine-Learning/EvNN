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

class EGRUCPUForwardTest(unittest.TestCase):
    def setUp(self):
        rnn = RNN_MAP['egru']
        self.seed = 5526

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)

        x = torch.rand(batch_size, time_steps, input_size, dtype=torch.float32)
        self.egru = rnn(input_size, hidden_size, zoneout=0.0, batch_first=True)

        self.x_cpu = x.clone()
        self.x_cpu_torch = self.x_cpu.detach().clone()

    def test_forward_y(self):
        with torch.no_grad():
            y1, _ = self.egru.forward(self.x_cpu)

            torch.manual_seed(self.seed)
            with mock.patch.object(self.egru, "use_custom_cuda", False):
                y2, _ = self.egru.forward(self.x_cpu_torch)

            assert torch.allclose(y1, y2)

    def test_forward_h(self):
        with torch.no_grad():
            _, (h1, _, _) = self.egru.forward(self.x_cpu)

            torch.manual_seed(self.seed)
            with mock.patch.object(self.egru, "use_custom_cuda", False):
                _, (h2, _, _) = self.egru.forward(self.x_cpu_torch)

            assert torch.allclose(h1, h2)

    def test_forward_o(self):
        with torch.no_grad():
            _, (_, o1, _) = self.egru.forward(self.x_cpu)

            torch.manual_seed(self.seed)
            with mock.patch.object(self.egru, "use_custom_cuda", False):
                _, (_, o2, _) = self.egru.forward(self.x_cpu_torch)

            assert torch.allclose(o1, o2)

    def test_forward_trace(self):
        with torch.no_grad():
            _, (_, _, t1) = self.egru.forward(self.x_cpu)

            torch.manual_seed(self.seed)
            with mock.patch.object(self.egru, "use_custom_cuda", False):
                _, (_, _, t2) = self.egru.forward(self.x_cpu_torch)

            assert torch.allclose(t1, t2)


class EGRUCUDABackwardTest(unittest.TestCase):
    def setUp(self):
        rnn = RNN_MAP['egru']
        self.seed = 5526

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)

        x = torch.rand(batch_size, time_steps, input_size, dtype=torch.float32)
        self.egru = rnn(input_size, hidden_size, zoneout=0.0, batch_first=True)

        self.x_cpu = x.clone()
        self.x_cpu_torch = self.x_cpu.detach().clone()
        self.x_cpu.requires_grad_(True)
        self.x_cpu_torch.requires_grad_(True)

    def test_backward_y(self):
        y1, _ = self.egru.forward(self.x_cpu)
        y1.backward(torch.ones_like(y1), retain_graph=True)

        torch.manual_seed(self.seed)
        
        y2, _ = self.egru.forward(self.x_cpu_torch)
        y2.backward(torch.ones_like(y2), retain_graph=True)

        assert torch.allclose(self.x_cpu.grad.data,
                              self.x_cpu_torch.grad.data, atol=1e-06)

    def test_backward_h(self):
        _, (h1, _, _) = self.egru.forward(self.x_cpu)
        h1.backward(torch.ones_like(h1), retain_graph=True)

        torch.manual_seed(self.seed)
        
        _, (h2, _, _) = self.egru.forward(self.x_cpu_torch)
        h2.backward(torch.ones_like(h2), retain_graph=True)

        assert torch.allclose(self.x_cpu.grad.data,
                              self.x_cpu_torch.grad.data, atol=1e-06)

    def test_backward_o(self):
        _, (_, o1, _) = self.egru.forward(self.x_cpu)
        o1.backward(torch.ones_like(o1), retain_graph=True)

        torch.manual_seed(self.seed)
        
        _, (_, o2, _) = self.egru.forward(self.x_cpu_torch)
        o2.backward(torch.ones_like(o2), retain_graph=True)

        assert torch.allclose(self.x_cpu.grad.data,
                              self.x_cpu_torch.grad.data, atol=1e-06)

    def test_backward_trace(self):
        _, (_, _, t1) = self.egru.forward(self.x_cpu)
        t1.backward(torch.ones_like(t1), retain_graph=True)

        torch.manual_seed(self.seed)
        
        _, (_, _, t2) = self.egru.forward(self.x_cpu_torch)
        t2.backward(torch.ones_like(t2), retain_graph=True)

        assert torch.allclose(self.x_cpu.grad.data,
                              self.x_cpu_torch.grad.data, atol=1e-06)


if __name__ == '__main__':
    unittest.main()

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
import data as d


def evaluate(model, eval_data, criterion, batch_size, bptt, ntokens, device, hidden_dims, return_hidden=False):
    # turn on evaluation mode
    model.eval()

    # initialize evaluation metrics
    iter_range = range(0, eval_data.size(0) - 1, bptt)

    total_loss = 0.
    mean_activities = torch.zeros(len(iter_range), dtype=torch.float16, device=device)
    layer_mean_activities = torch.zeros((len(iter_range), model.nlayers), dtype=torch.float16, device=device)
    centered_cell_states = [torch.zeros((len(iter_range), batch_size, hidden_dim), dtype=torch.float16, device=device)
                            for hidden_dim in hidden_dims]

    if return_hidden:
        all_hidden = [torch.zeros((batch_size, eval_data.size(0), hidden_dim), dtype=torch.float16, device=device)
                      for hidden_dim in hidden_dims]

    # run evaluation, no gradients required
    with torch.no_grad():

        # initialize hidden states
        hidden = model.init_hidden(batch_size)

        # iterate evaluation data
        for num_iter, index in enumerate(iter_range):

            # draw a batch
            data, targets = d.get_batch(eval_data, index, seq_len=bptt, batch_first=True)

            # run model
            output, hidden, hid_full, hid_dropped = model(data, hidden)

            # record evaluation metrics
            output_gate_vals = [torch.where(hid == 0, torch.zeros_like(hid), torch.ones_like(hid)).to(device) for hid in
                                hid_full]

            layer_mean_activity = torch.tensor([torch.mean(ogv).to(device) for ogv in output_gate_vals])
            layer_mean_activities[num_iter] = layer_mean_activity

            mean_activity = torch.mean(torch.cat([ogv.flatten() for ogv in output_gate_vals]))
            mean_activities[num_iter] = mean_activity

            if model.rnn_type == 'egrud':
                for l in range(model.nlayers):
                    rnn = model.rnns[l]
                    centered_cell_states[l][num_iter] = hidden[l][0].squeeze() - rnn.thr.detach()

            # record hidden states if return_hidden is True
            if return_hidden:
                for k, h in enumerate(hid_full):
                    all_hidden[k][:, index:index+data.size(1), :] = h

            # compute loss
            output_flat = output.view(-1, ntokens)
            total_loss += data.numel() * criterion(output_flat, targets).item()

    if return_hidden:
        return total_loss / eval_data.numel(), torch.mean(mean_activities), \
               torch.mean(layer_mean_activities, dim=0), \
               torch.std(layer_mean_activities, dim=0), \
               centered_cell_states, \
               all_hidden
    else:
        return total_loss / eval_data.numel(), torch.mean(mean_activities), \
               torch.mean(layer_mean_activities, dim=0), \
               torch.std(layer_mean_activities, dim=0), \
               centered_cell_states

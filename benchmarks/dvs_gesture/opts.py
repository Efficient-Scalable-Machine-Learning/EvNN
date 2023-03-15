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

import argparse
import json

from egru import RNNType


def parse_opts():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=3000)
    argparser.add_argument(
        '--data', type=str, required=True, help='path to datasets')
    argparser.add_argument('--cache', type=str,
                           required=True, help='path to temp cache')
    argparser.add_argument('--logdir', type=str,
                           required=True, help='scratch directory for jobs')
    argparser.add_argument('--resume-path', type=str,
                           required=False, help='Resume training')
    argparser.add_argument('--run-title', type=str, required=False, default='')
    argparser.add_argument('--log-interval', type=int, default=50)
    argparser.add_argument('--batch-size', type=int, default=256)
    argparser.add_argument('--frame-size', type=int, default=64)
    argparser.add_argument('--frame-time', type=int, default=25,
                           help='Time in ms to collect events into each frame')
    argparser.add_argument('--event-agg-method', type=str, default='bool',
                           choices=['mean', 'diff', 'bool', 'none'])
    argparser.add_argument('--flatten-frame', action='store_true')
    argparser.add_argument('--use-cnn', action='store_true')
    argparser.add_argument('--augment-data', action='store_true')
    argparser.add_argument('--learning-rate', type=float, default=0.001)
    argparser.add_argument('--lr-gamma', type=float, default=0.8)
    argparser.add_argument('--lr-decay-epochs', type=int, default=100)
    argparser.add_argument('--use-rmsprop', action='store_true')
    argparser.add_argument('--use-grad-clipping', action='store_true')
    argparser.add_argument('--grad-clip-norm', type=float, default=2.0)
    argparser.add_argument('--rnn-type', type=str,
                           default='lstm', choices=[e.value for e in RNNType])
    argparser.add_argument('--units', type=int, default=256)
    argparser.add_argument('--unit-size', type=int, default=1)
    argparser.add_argument('--num-layers', type=int, default=1)
    argparser.add_argument('--use-all-timesteps', action='store_true')
    argparser.add_argument('--train-epochs', type=int, default=100)
    argparser.add_argument('--activity-regularization', action='store_true')
    argparser.add_argument(
        '--activity-regularization-constant', type=float, default=1.)
    argparser.add_argument(
        '--activity-regularization-target', type=float, default=0.05)
    argparser.add_argument(
        '--voltage-regularization-target', type=float, default=0.05)
    argparser.add_argument('--dropconnect', type=float, default=0.0)
    argparser.add_argument('--dropout', type=float, default=0.0)
    argparser.add_argument('--zoneout', type=float, default=0.0)
    argparser.add_argument('--pseudo-derivative-width',
                           type=float, default=1.0)
    argparser.add_argument('--threshold-mean', type=float, default=0.0)
    args = argparser.parse_args()

    return args


def store_args(file, args):
    with open(file, 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def load_args(file):
    parser = argparse.ArgumentParser()
    args = parser.parse_args('')
    with open(file, 'r') as f:
        args.__dict__ = json.load(f)

    return args

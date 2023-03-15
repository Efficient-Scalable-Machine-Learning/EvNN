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

import random
import time
from datetime import datetime
from enum import Enum


class Timer:
    def __init__(self):
        self._startime = None
        self._endtime = None
        self.difftime = None

    def __enter__(self):
        self._startime = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._endtime = time.time()
        self.difftime = self._endtime - self._startime


class RNNType(Enum):
    LSTM = 'lstm'
    GRU = 'gru'
    EGRU = 'egru'


class Optimizer(Enum):
    adam = 'adam'
    sgd = 'sgd'


def get_random_name(prefix='baseline'):
    datetime_suffix = datetime.now().strftime("D%d-%m-%Y-T%H-%M-%S")
    randnum = str(random.randint(1e3, 1e5))
    sim_name = f"{prefix}-{randnum}-{datetime_suffix}"
    return sim_name

// Copyright (c) 2023  Khaleelulla Khan Nazeer
// This file incorporates work covered by the following copyright:
// Copyright 2020 LMNT, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#pragma once

// GENERAL NOTES:
// No pointers may be null unless otherwise specified.
// All pointers are expected to point to device memory.
// The square brackets below describe tensor shapes, where
//     T = number of RNN time steps
//     N = batch size
//     C = input size
//     H = hidden size
// and the rightmost dimension changes the fastest.

#ifdef __NVCC__
#include "evnn/egru.h"
#else
#ifdef WITH_CUDA
#include "evnn/egru.h"
#endif
#endif
#include "evnn/egru_cpu.h"

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
#include <fstream>
#include <iostream>

#ifdef WITH_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include <torch/extension.h>
#include <vector>

#include "evnn.h"
#include "support.h"

namespace
{

#ifdef WITH_CUDA
  using evnn::v0::egru::BackwardPass;
  using evnn::v0::egru::ForwardPass;
#endif
  using evnn::v0::egru::BackwardPassCPU;
  using evnn::v0::egru::ForwardPassCPU;

  using torch::Tensor;

#ifdef WITH_CUDA
  std::vector<Tensor> egru_forward(
      bool training,
      float zoneout_prob,
      Tensor x,
      Tensor y0,
      Tensor kernel,
      Tensor recurrent_kernel,
      Tensor bias,
      Tensor recurrent_bias,
      Tensor thr,
      Tensor zoneout_mask)
  {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto input_size = x.size(2);
    const auto hidden_size = recurrent_kernel.size(0);
    const bool has_zoneout = zoneout_prob && zoneout_mask.size(0);

    CHECK_INPUT(x);
    CHECK_INPUT(y0);
    CHECK_INPUT(kernel);
    CHECK_INPUT(recurrent_kernel);
    CHECK_INPUT(bias);
    CHECK_INPUT(recurrent_bias);
    CHECK_INPUT(thr);
    CHECK_INPUT(zoneout_mask);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor h = torch::zeros({time_steps + 1, batch_size, hidden_size}, options);
    Tensor output = torch::zeros({time_steps + 1, batch_size, hidden_size}, options);
    Tensor output_gate = torch::zeros({time_steps + 1, batch_size, hidden_size}, options);
    Tensor cache = torch::zeros({time_steps, batch_size, hidden_size * 5}, options);
    Tensor tmp_Wx = torch::zeros({time_steps, batch_size, hidden_size * 3}, options);
    Tensor tmp_Rh = torch::zeros({batch_size, hidden_size * 3}, options);

    Tensor trace = torch::zeros({time_steps + 1, batch_size, hidden_size}, options);

    output[0] = y0;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "egru_forward", ([&]
                                                                          {
    ForwardPass<typename native_type<scalar_t>::T> forward(
        training,
        batch_size,
        input_size,
        hidden_size,
        at::cuda::getCurrentCUDABlasHandle(),
        at::cuda::getCurrentCUDAStream());

    forward.Run(
        time_steps,
        ptr<scalar_t>(kernel),
        ptr<scalar_t>(recurrent_kernel),
        ptr<scalar_t>(bias),
        ptr<scalar_t>(recurrent_bias),
        ptr<scalar_t>(x),
        ptr<scalar_t>(h),
        ptr<scalar_t>(output),
        ptr<scalar_t>(cache),
        ptr<scalar_t>(output_gate),
        ptr<scalar_t>(thr),
        ptr<scalar_t>(tmp_Wx),
        ptr<scalar_t>(tmp_Rh),
        ptr<scalar_t>(trace),
        has_zoneout ? zoneout_prob : 0.0f,
        has_zoneout ? ptr<scalar_t>(zoneout_mask) : nullptr); }));

    return {output, cache, h, output_gate, trace};
  }

  std::vector<Tensor> egru_backward(
      Tensor x_t,
      Tensor kernel_t,
      Tensor recurrent_kernel_t,
      Tensor bias,
      Tensor recurrent_bias,
      Tensor thr,
      Tensor zoneout_mask,
      Tensor dampening_factor,
      Tensor pseudo_derivative_support,
      Tensor max_grad_norm,
      Tensor y,
      Tensor h,
      Tensor cache,
      Tensor dy_new,
      Tensor dh_new,
      Tensor dout_gate,
      Tensor dtrs)
  {
    const auto input_size = x_t.size(0);
    const auto time_steps = x_t.size(1);
    const auto batch_size = x_t.size(2);
    const auto hidden_size = recurrent_kernel_t.size(1);
    const bool has_zoneout = !!zoneout_mask.size(0);

    CHECK_INPUT(x_t);
    CHECK_INPUT(kernel_t);
    CHECK_INPUT(recurrent_kernel_t);
    CHECK_INPUT(bias);
    CHECK_INPUT(recurrent_bias);
    CHECK_INPUT(thr);
    CHECK_INPUT(dampening_factor);
    CHECK_INPUT(pseudo_derivative_support);
    CHECK_INPUT(h);
    CHECK_INPUT(y);
    CHECK_INPUT(cache);
    CHECK_INPUT(dy_new);
    CHECK_INPUT(dh_new);
    CHECK_INPUT(dout_gate);
    CHECK_INPUT(dtrs);
    CHECK_INPUT(zoneout_mask);

    const auto options = x_t.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    Tensor dx = torch::zeros({time_steps, batch_size, input_size}, options);
    Tensor dW = torch::zeros({input_size, hidden_size * 3}, options);
    Tensor dR = torch::zeros({hidden_size, hidden_size * 3}, options);
    Tensor dbx = torch::zeros({hidden_size * 3}, options);
    Tensor dbr = torch::zeros({hidden_size * 3}, options);
    Tensor dthr = torch::zeros({hidden_size}, options);
    Tensor dy = torch::zeros({batch_size, hidden_size}, options);
    Tensor dh = torch::zeros({batch_size, hidden_size}, options);
    Tensor dp = torch::zeros({time_steps, batch_size, hidden_size * 3}, options);
    Tensor dq = torch::zeros({time_steps, batch_size, hidden_size * 3}, options);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_t.scalar_type(), "egru_backward", ([&]
                                                                             {
    BackwardPass<typename native_type<scalar_t>::T> backward(
        batch_size,
        input_size,
        hidden_size,
        at::cuda::getCurrentCUDABlasHandle(),
        at::cuda::getCurrentCUDAStream());

    backward.Run(
        time_steps,
        ptr<scalar_t>(dampening_factor),
        ptr<scalar_t>(pseudo_derivative_support),
        ptr<scalar_t>(kernel_t),
        ptr<scalar_t>(recurrent_kernel_t),
        ptr<scalar_t>(bias),
        ptr<scalar_t>(recurrent_bias),
        ptr<scalar_t>(thr),
        ptr<scalar_t>(x_t),
        ptr<scalar_t>(y),
        ptr<scalar_t>(h),
        ptr<scalar_t>(cache),
        ptr<scalar_t>(dy_new),
        ptr<scalar_t>(dh_new),
        ptr<scalar_t>(dout_gate),
        ptr<scalar_t>(dx),
        ptr<scalar_t>(dW),
        ptr<scalar_t>(dR),
        ptr<scalar_t>(dbx),
        ptr<scalar_t>(dbr),
        ptr<scalar_t>(dthr),
        ptr<scalar_t>(dy),
        ptr<scalar_t>(dh),
        ptr<scalar_t>(dtrs),
        ptr<scalar_t>(dp),
        ptr<scalar_t>(dq),
        has_zoneout ? ptr<scalar_t>(zoneout_mask) : nullptr); }));

    auto clip_coef_clamped = torch::ones({1}, options);
    if (max_grad_norm.item<double>() > 0.0)
    {
      Tensor norms = torch::empty({7}, options);
      auto index = 0;
      for (auto &grad : {&dy, &dW, &dR, &dbx, &dbr, &dthr})
      {
        norms[index] = torch::norm(*grad);
        index++;
      }
      auto total_norm = torch::norm(norms);
      const auto clip_coef = max_grad_norm / (total_norm + 1e-6);
      clip_coef_clamped = torch::clamp(clip_coef, 0.0, 1.0);
      // std::cout << total_norm << std::endl;
      // if(clip_coef_clamped.item<double>() < 1.0){
      //   std::cout << clip_coef_clamped << std::endl;
      // }
    }

    return {dx,
            dy * clip_coef_clamped,
            dW * clip_coef_clamped,
            dR * clip_coef_clamped,
            dbx * clip_coef_clamped,
            dbr * clip_coef_clamped,
            dthr * clip_coef_clamped,
            dp, dq,
            clip_coef_clamped};
  }

#endif

  std::vector<Tensor> egru_forward_cpu(
      bool training,
      float zoneout_prob,
      Tensor x,
      Tensor y0,
      Tensor kernel,
      Tensor recurrent_kernel,
      Tensor bias,
      Tensor recurrent_bias,
      Tensor thr,
      Tensor zoneout_mask)
  {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto input_size = x.size(2);
    const auto hidden_size = recurrent_kernel.size(0);
    const bool has_zoneout = zoneout_prob && zoneout_mask.size(0);

    CHECK_CPU_INPUT(x);
    CHECK_CPU_INPUT(y0);
    CHECK_CPU_INPUT(kernel);
    CHECK_CPU_INPUT(recurrent_kernel);
    CHECK_CPU_INPUT(bias);
    CHECK_CPU_INPUT(recurrent_bias);
    CHECK_CPU_INPUT(thr);
    CHECK_CPU_INPUT(zoneout_mask);

    const auto options = x.options();
    Tensor h = torch::zeros({time_steps + 1, batch_size, hidden_size}, options);
    Tensor output = torch::zeros({time_steps + 1, batch_size, hidden_size}, options);
    Tensor output_gate = torch::zeros({time_steps + 1, batch_size, hidden_size}, options);
    Tensor cache = torch::zeros({time_steps, batch_size, hidden_size * 5}, options);
    Tensor tmp_Wx = torch::zeros({time_steps, batch_size, hidden_size * 3}, options);
    Tensor tmp_Rh = torch::zeros({batch_size, hidden_size * 3}, options);

    Tensor trace = torch::zeros({time_steps + 1, batch_size, hidden_size}, options);

    output[0] = y0;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "egru_forward_cpu", ([&]
                                                                     {
    ForwardPassCPU<typename native_type<scalar_t>::T> forward(
        training,
        batch_size,
        input_size,
        hidden_size);

    forward.Run(
        time_steps,
        ptr<scalar_t>(kernel),
        ptr<scalar_t>(recurrent_kernel),
        ptr<scalar_t>(bias),
        ptr<scalar_t>(recurrent_bias),
        ptr<scalar_t>(x),
        ptr<scalar_t>(h),
        ptr<scalar_t>(output),
        ptr<scalar_t>(cache),
        ptr<scalar_t>(output_gate),
        ptr<scalar_t>(thr),
        ptr<scalar_t>(tmp_Wx),
        ptr<scalar_t>(tmp_Rh),
        ptr<scalar_t>(trace),
        has_zoneout ? zoneout_prob : 0.0f,
        has_zoneout ? ptr<scalar_t>(zoneout_mask) : nullptr); }));

    return {output, cache, h, output_gate, trace};
  }

  std::vector<Tensor> egru_backward_cpu(
      Tensor x_t,
      Tensor kernel_t,
      Tensor recurrent_kernel_t,
      Tensor bias,
      Tensor recurrent_bias,
      Tensor thr,
      Tensor zoneout_mask,
      Tensor dampening_factor,
      Tensor pseudo_derivative_support,
      Tensor max_grad_norm,
      Tensor y,
      Tensor h,
      Tensor cache,
      Tensor dy_new,
      Tensor dh_new,
      Tensor dout_gate,
      Tensor dtrs)
  {
    const auto input_size = x_t.size(0);
    const auto time_steps = x_t.size(1);
    const auto batch_size = x_t.size(2);
    const auto hidden_size = recurrent_kernel_t.size(1);
    const bool has_zoneout = !!zoneout_mask.size(0);

    CHECK_CPU_INPUT(x_t);
    CHECK_CPU_INPUT(kernel_t);
    CHECK_CPU_INPUT(recurrent_kernel_t);
    CHECK_CPU_INPUT(bias);
    CHECK_CPU_INPUT(recurrent_bias);
    CHECK_CPU_INPUT(thr);
    CHECK_CPU_INPUT(dampening_factor);
    CHECK_CPU_INPUT(pseudo_derivative_support);
    CHECK_CPU_INPUT(h);
    CHECK_CPU_INPUT(y);
    CHECK_CPU_INPUT(cache);
    CHECK_CPU_INPUT(dy_new);
    CHECK_CPU_INPUT(dh_new);
    CHECK_CPU_INPUT(dout_gate);
    CHECK_CPU_INPUT(dtrs);
    CHECK_CPU_INPUT(zoneout_mask);

    const auto options = x_t.options();
    Tensor dx = torch::zeros({time_steps, batch_size, input_size}, options);
    Tensor dW = torch::zeros({input_size, hidden_size * 3}, options);
    Tensor dR = torch::zeros({hidden_size, hidden_size * 3}, options);
    Tensor dbx = torch::zeros({hidden_size * 3}, options);
    Tensor dbr = torch::zeros({hidden_size * 3}, options);
    Tensor dthr = torch::zeros({hidden_size}, options);
    Tensor dy = torch::zeros({batch_size, hidden_size}, options);
    Tensor dh = torch::zeros({batch_size, hidden_size}, options);
    Tensor dp = torch::zeros({time_steps, batch_size, hidden_size * 3}, options);
    Tensor dq = torch::zeros({time_steps, batch_size, hidden_size * 3}, options);

    AT_DISPATCH_FLOATING_TYPES(x_t.scalar_type(), "egru_backward_cpu", ([&]
                                                                        {
    BackwardPassCPU<typename native_type<scalar_t>::T> backward(
        batch_size,
        input_size,
        hidden_size);

    backward.Run(
        time_steps,
        ptr<scalar_t>(dampening_factor),
        ptr<scalar_t>(pseudo_derivative_support),
        ptr<scalar_t>(kernel_t),
        ptr<scalar_t>(recurrent_kernel_t),
        ptr<scalar_t>(bias),
        ptr<scalar_t>(recurrent_bias),
        ptr<scalar_t>(thr),
        ptr<scalar_t>(x_t),
        ptr<scalar_t>(y),
        ptr<scalar_t>(h),
        ptr<scalar_t>(cache),
        ptr<scalar_t>(dy_new),
        ptr<scalar_t>(dh_new),
        ptr<scalar_t>(dout_gate),
        ptr<scalar_t>(dx),
        ptr<scalar_t>(dW),
        ptr<scalar_t>(dR),
        ptr<scalar_t>(dbx),
        ptr<scalar_t>(dbr),
        ptr<scalar_t>(dthr),
        ptr<scalar_t>(dy),
        ptr<scalar_t>(dh),
        ptr<scalar_t>(dtrs),
        ptr<scalar_t>(dp),
        ptr<scalar_t>(dq),
        has_zoneout ? ptr<scalar_t>(zoneout_mask) : nullptr); }));

    auto clip_coef_clamped = torch::ones({1}, options);
    if (max_grad_norm.item<double>() > 0.0)
    {
      Tensor norms = torch::empty({7}, options);
      auto index = 0;
      for (auto &grad : {&dy, &dW, &dR, &dbx, &dbr, &dthr})
      {
        norms[index] = torch::norm(*grad);
        index++;
      }
      auto total_norm = torch::norm(norms);
      const auto clip_coef = max_grad_norm / (total_norm + 1e-6);
      clip_coef_clamped = torch::clamp(clip_coef, 0.0, 1.0);
      // std::cout << total_norm << std::endl;
      // if(clip_coef_clamped.item<double>() < 1.0){
      //   std::cout << clip_coef_clamped << std::endl;
      // }
    }

    return {dx,
            dy * clip_coef_clamped,
            dW * clip_coef_clamped,
            dR * clip_coef_clamped,
            dbx * clip_coef_clamped,
            dbr * clip_coef_clamped,
            dthr * clip_coef_clamped,
            dp, dq,
            clip_coef_clamped};
  }

} // anonymous namespace

void egru_init(py::module &m)
{
  #ifdef WITH_CUDA
  m.def("egru_forward", &egru_forward, "EGRU forward", py::call_guard<py::gil_scoped_release>());
  m.def("egru_backward", &egru_backward, "EGRU backward", py::call_guard<py::gil_scoped_release>());
  #endif
  m.def("egru_forward_cpu", &egru_forward_cpu, "EGRU forward (CPU)", py::call_guard<py::gil_scoped_release>());
  m.def("egru_backward_cpu", &egru_backward_cpu, "EGRU backward (CPU)", py::call_guard<py::gil_scoped_release>());
}

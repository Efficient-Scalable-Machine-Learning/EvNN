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

#include <iostream>
#include "blas.h"
#include "evnn/egru_cpu.h"
#include "inline_ops.h"

namespace
{

  template <typename T, bool ApplyZoneout>
  void PointwiseOperations(const int row, // hidden
                           const int col, // batch
                           const int batch_dim,
                           const int hidden_dim,
                           const T *h_tm1,
                           const T *h,
                           const T *thr,
                           const T *v,
                           const T *dy_new,
                           const T *dh_new,
                           const T *dout_gate,
                           const T *dtrs,
                           T *dbx_out,
                           T *dbr_out,
                           T *dthr_out,
                           T *dy_inout,
                           T *dh_inout,
                           T *dtrs_inout,
                           T *dp_out,
                           T *dq_out,
                           const T *dampening_factor,
                           const T *pseudo_derivative_support,
                           const T *zoneout_mask)
  { // Zoneout mask (only used if ApplyZoneout==true)

    if (row >= hidden_dim || col >= batch_dim)
      return;

    const int base_idx = col * hidden_dim + row;

    dtrs_inout[base_idx] += static_cast<T>(0.9) * dtrs[base_idx];
    T dy_total = dy_new[base_idx] + dy_inout[base_idx] + static_cast<T>(0.1) * dtrs[base_idx];

    const int stride4_base_idx = col * (hidden_dim * 5) + row;
    const int z_idx = stride4_base_idx + 0 * hidden_dim;
    const int r_idx = stride4_base_idx + 1 * hidden_dim;
    const int g_idx = stride4_base_idx + 2 * hidden_dim;
    const int q_g_idx = stride4_base_idx + 3 * hidden_dim;
    const int spk_inp_idx = stride4_base_idx + 4 * hidden_dim;

    const T z = v[z_idx];
    const T r = v[r_idx];
    const T g = v[g_idx];
    const T q_g = v[q_g_idx];
    const T spk_inp = v[spk_inp_idx];

    T dh_total = dh_inout[base_idx] + dh_new[base_idx];
    // if (row == 0 || col == 0)
    //   std::cout << "d_total" << dh_total << std::endl;
    const T dout = dout_gate[base_idx] + (h[base_idx] + thr[row] * heaviside(spk_inp)) * dy_total + (static_cast<T>(-1.0) * thr[row] * dh_total);
    const T d_heavi = *dampening_factor * d_heaviside(spk_inp, *pseudo_derivative_support) * dout;
    const T dthr = static_cast<T>(-1.0) * (d_heavi + heaviside(spk_inp) * dh_total);
    dh_total = heaviside(spk_inp) * dy_total + d_heavi + dh_total;
    dy_inout[base_idx] = static_cast<T>(0.0);

    // if (row == 0 || col == 0)
    //   std::cout << "d_heavi " << d_heavi << std::endl;
    //   std::cout << "dout " << dout << std::endl;
    //   std::cout << "dthr " << dthr << std::endl;
    //   std::cout << "dy_total " << dy_total << std::endl;

    if (ApplyZoneout)
    {
      const T mask = zoneout_mask[base_idx];
      dh_inout[base_idx] = (static_cast<T>(1.0) - mask) * dh_total;
      dh_total = mask * dh_total;
      dh_inout[base_idx] += z * dh_total;
    }
    else
    {
      dh_inout[base_idx] = z * dh_total; // - (heaviside(spk_inp) + h[base_idx] * d_heavi);
    }

    const T dg = (static_cast<T>(1.0) - z) * dh_total;
    const T dz = (h_tm1[base_idx] - g) * dh_total;
    const T dp_g = d_tanh(g) * dg;
    const T dq_g = dp_g * r;
    const T dr = dp_g * q_g;
    const T dp_r = d_sigmoid(r) * dr;
    const T dq_r = dp_r;
    const T dp_z = d_sigmoid(z) * dz;
    const T dq_z = dp_z;

    const int idx = col * (hidden_dim * 3) + row;

    dp_out[idx + 0 * hidden_dim] = dp_z;
    dp_out[idx + 1 * hidden_dim] = dp_r;
    dp_out[idx + 2 * hidden_dim] = dp_g;

    dq_out[idx + 0 * hidden_dim] = dq_z;
    dq_out[idx + 1 * hidden_dim] = dq_r;
    dq_out[idx + 2 * hidden_dim] = dq_g;

    dbx_out[row + 0 * hidden_dim] += dp_z;
    dbx_out[row + 1 * hidden_dim] += dp_r;
    dbx_out[row + 2 * hidden_dim] += dp_g;

    dbr_out[row + 0 * hidden_dim] += dq_z;
    dbr_out[row + 1 * hidden_dim] += dq_r;
    dbr_out[row + 2 * hidden_dim] += dq_g;

    dthr_out[row] += dthr;
  }

} // anonymous namespace

namespace evnn
{
  namespace v0
  {
    namespace egru
    {

      template <typename T>
      struct BackwardPassCPU<T>::private_data
      {
        int batch_size;
        int input_size;
        int hidden_size;
      };

      template <typename T>
      BackwardPassCPU<T>::BackwardPassCPU(
          const int batch_size,
          const int input_size,
          const int hidden_size) : data_(new private_data)
      {
        data_->batch_size = batch_size;
        data_->input_size = input_size;
        data_->hidden_size = hidden_size;
      }

      template <typename T>
      BackwardPassCPU<T>::~BackwardPassCPU()
      {
        delete data_;
      }

      template <typename T>
      void BackwardPassCPU<T>::Iterate(
          const T *W_t, // [H*3,C]
          const T *R_t, // [H*3,H]
          const T *bx,  // [H*3]
          const T *br,  // [H*3]
          const T *thr,
          const T *x_t, // [C,N]
          const T *y,
          const T *h,      // [N,H]
          const T *v,      // [N,H*4]
          const T *dy_new, // [N,H]
          const T *dh_new, // [N,H]
          const T *dout_gate,
          const T *dtrs,
          T *dx,  // [N,C]
          T *dW,  // [C,H*3]
          T *dR,  // [H,H*3]
          T *dbx, // [H*3]
          T *dbr, // [H*3]
          T *dthr,
          T *dy,
          T *dh, // [N,H]
          T *dtrs_inout,
          T *dp, // [N,H*3]
          T *dq, // [N,H*3]
          const T *dampening_factor,
          const T *pseudo_derivative_support,
          const T *zoneout_mask)
      { // [N,H]

        const T alpha = static_cast<T>(1.0);
        const T beta_sum = static_cast<T>(1.0);
        const T beta_assign = static_cast<T>(0.0);

        const int batch_size = data_->batch_size;
        const int hidden_size = data_->hidden_size;
        const int input_size = data_->input_size;

        IterateInternal(
            R_t,
            thr,
            y,
            h,
            v,
            dy_new,
            dh_new,
            dout_gate,
            dtrs,
            dbx,
            dbr,
            dthr,
            dy,
            dh,
            dtrs_inout,
            dp,
            dq,
            dampening_factor,
            pseudo_derivative_support,
            zoneout_mask);

        c_blas<T>::gemm(CblasColMajor,
                        CblasNoTrans, CblasNoTrans,
                        hidden_size * 3, input_size, batch_size,
                        alpha,
                        dp, hidden_size * 3,
                        x_t, batch_size,
                        beta_sum,
                        dW, hidden_size * 3);

        // Wait for pointwise operations to complete since there's a
        // data dependency between its output (`dp`, `dq`) and the following matmuls.

        c_blas<T>::gemm(CblasColMajor,
                        CblasNoTrans, CblasNoTrans,
                        input_size, batch_size, hidden_size * 3,
                        alpha,
                        W_t, input_size,
                        dp, hidden_size * 3,
                        beta_assign,
                        dx, input_size);

        c_blas<T>::gemm(CblasColMajor,
                        CblasNoTrans, CblasTrans,
                        hidden_size * 3, hidden_size, batch_size,
                        alpha,
                        dq, hidden_size * 3,
                        y, hidden_size,
                        beta_sum,
                        dR, hidden_size * 3);
      }

      template <typename T>
      void BackwardPassCPU<T>::IterateInternal(
          const T *R_t, // [H*3,H]
          const T *thr,
          const T *h_tm1,
          const T *h,      // [N,H]
          const T *v,      // [N,H*4]
          const T *dy_new, // [N,H]
          const T *dh_new, // [N,H]
          const T *dout_gate,
          const T *dtrs,
          T *dbx, // [H*3]
          T *dbr, // [H*3]
          T *dthr,
          T *dy,
          T *dh, // [N,H]
          T *dtrs_inout,
          T *dp, // [N,H*3]
          T *dq, // [N,H*3]
          const T *dampening_factor,
          const T *pseudo_derivative_support,
          const T *zoneout_mask)
      { // [N,H]
        const T alpha = static_cast<T>(1.0);
        const T beta_sum = static_cast<T>(1.0);

        const int batch_size = data_->batch_size;
        const int hidden_size = data_->hidden_size;

        for (int row = 0; row < hidden_size; ++row)
        {
          for (int col = 0; col < batch_size; ++col)
          {
            if (zoneout_mask)
            {
              PointwiseOperations<T, true>(
                  row,
                  col,
                  batch_size,
                  hidden_size,
                  h_tm1,
                  h,
                  thr,
                  v,
                  dy_new,
                  dh_new,
                  dout_gate,
                  dtrs,
                  dbx,
                  dbr,
                  dthr,
                  dy,
                  dh,
                  dtrs_inout,
                  dp,
                  dq,
                  dampening_factor,
                  pseudo_derivative_support,
                  zoneout_mask);
            }
            else
            {
              PointwiseOperations<T, false>(
                  row,
                  col,
                  batch_size,
                  hidden_size,
                  h_tm1,
                  h,
                  thr,
                  v,
                  dy_new,
                  dh_new,
                  dout_gate,
                  dtrs,
                  dbx,
                  dbr,
                  dthr,
                  dy,
                  dh,
                  dtrs_inout,
                  dp,
                  dq,
                  dampening_factor,
                  pseudo_derivative_support,
                  nullptr);
            }
          }
        }

        c_blas<T>::gemm(CblasColMajor,
                        CblasNoTrans, CblasNoTrans,
                        hidden_size, batch_size, hidden_size * 3,
                        alpha,
                        R_t, hidden_size,
                        dq, hidden_size * 3,
                        beta_sum,
                        dy, hidden_size);
      }

      template <typename T>
      void BackwardPassCPU<T>::Run(
          const int steps,
          const T *dampening_factor,
          const T *pseudo_derivative_support,
          const T *W_t,
          const T *R_t,
          const T *bx,
          const T *br,
          const T *thr,
          const T *x_t,
          const T *y,
          const T *h,
          const T *v,
          const T *dy_new,
          const T *dh_new,
          const T *dout_gate,
          T *dx,
          T *dW,
          T *dR,
          T *dbx,
          T *dbr,
          T *dthr,
          T *dy,
          T *dh,
          T *dtrs,
          T *dp,
          T *dq,
          const T *zoneout_mask)
      {

        const T alpha = static_cast<T>(1.0);
        const T beta_sum = static_cast<T>(1.0);
        const T beta_assign = static_cast<T>(0.0);

        const int batch_size = data_->batch_size;
        const int input_size = data_->input_size;
        const int hidden_size = data_->hidden_size;

        const int NH = batch_size * hidden_size;
        for (int i = steps - 1; i >= 0; --i)
        {
          IterateInternal(
              R_t,
              thr,
              h + i * NH,
              h + (i + 1) * NH,
              v + i * NH * 5,
              dy_new + (i + 1) * NH,
              dh_new + (i + 1) * NH,
              dout_gate + (i + 1) * NH,
              dtrs + (i + 1) * NH,
              dbx,
              dbr,
              dthr,
              dy,
              dh,
              dtrs + (i)*NH,
              dp + i * NH * 3,
              dq + i * NH * 3,
              dampening_factor,
              pseudo_derivative_support,
              zoneout_mask ? zoneout_mask + i * NH : nullptr);
        }

        c_blas<T>::gemm(CblasColMajor,
                        CblasNoTrans, CblasNoTrans,
                        input_size, batch_size * steps, hidden_size * 3,
                        alpha,
                        W_t, input_size,
                        dp, hidden_size * 3,
                        beta_assign,
                        dx, input_size);

        c_blas<T>::gemm(CblasColMajor,
                        CblasNoTrans, CblasTrans,
                        hidden_size * 3, hidden_size, batch_size * steps,
                        alpha,
                        dq, hidden_size * 3,
                        y, hidden_size,
                        beta_sum,
                        dR, hidden_size * 3);

        c_blas<T>::gemm(CblasColMajor,
                        CblasNoTrans, CblasNoTrans,
                        hidden_size * 3, input_size, batch_size * steps,
                        alpha,
                        dp, hidden_size * 3,
                        x_t, batch_size * steps,
                        beta_sum,
                        dW, hidden_size * 3);
      }

      // template struct BackwardPassCPU<half>;
      template struct BackwardPassCPU<float>;
      template struct BackwardPassCPU<double>;

    } // namespace egru
  }   // namespace v0
} // namespace evnn

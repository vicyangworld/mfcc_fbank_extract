/*
 * Copyright (C) 2018 Arm Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __KWS_MFCC_H__
#define __KWS_MFCC_H__

#define SAMP_FREQ 16000
#define NUM_FBANK_BINS 40
#define MEL_LOW_FREQ 100
#define MEL_HIGH_FREQ 7800

#define M_2PI 6.283185307179586476925286766559005
#include <stdlib.h>
#include <math.h>
#include <vector>
#include "kiss_fftr.h"
#include <ctime>
#include <string>

using namespace std;
namespace KWS {

class MFCC{
  private:
    int num_mfcc_features;
    int frame_len;
    int frame_len_padded;
    int mfcc_dec_bits;
    kiss_fftr_cfg kiss_fftr_state_;
    kiss_fft_cpx * cframe_;
    bool _is_dct;
    string & _win_type;
    bool _add_rand_noise;

    float * frame;
    float * energy;
    float * mel_energies;
    float * window_func;
    int32_t * fbank_filter_first;
    int32_t * fbank_filter_last;
    float ** mel_fbank;
    float * dct_matrix;
    float * rand_noise;

    float * create_dct_matrix(int32_t input_length, int32_t coefficient_count); 
    float ** create_mel_fbank();
    float *  _create_window(int);
    float *  _create_rand_noise(int len);
 
    static inline float InverseMelScale(float mel_freq) {
      return 700.0f * (expf (mel_freq / 1127.0f) - 1.0f);
    }

    static inline float MelScale(float freq) {
      return 1127.0f * logf (1.0f + freq / 700.0f);
    }

    inline float randUniform() {
      return (rand() + 1.0) / (RAND_MAX+2.0);
    }

    inline float _randGauss() {
      return sqrtf(-2 * logf(randUniform())) * cosf(2*M_2PI*randUniform());
    }

    void _preEmphasize(float *frame, int frame_len, float preemph_coeff); 
    void mfcc_compute_scaled(float * mfcc_out);
    void fbank_compute_scaled(float * fbank_out);
    void fbank_compute_kaldilike(float * fbank_out);
  public:
    //mfcc_dec_bits for quantization. currently not used.
    MFCC(int num_mfcc_features, int frame_len, int mfcc_dec_bits, string win_type, bool is_dct=true , bool add_rand_noise=true) ;
    ~MFCC();
    void mfcc_compute(const int16_t* data, float* mfcc_out);
    void mfcc_compute(const float * data, float * mfcc_out);
    void fbank_compute(const float * audio_data, bool kaldi_like, float* fbank_out);
};
}
#endif

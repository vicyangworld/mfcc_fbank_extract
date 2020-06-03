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

/*
 * Description: MFCC feature extraction to match with TensorFlow MFCC Op
 */

#include <string.h>
#include "mfcc.h"
#include "float.h"
#include <vector>
using namespace std;

namespace KWS {

MFCC::MFCC(int num_mfcc_features, int frame_len, int mfcc_dec_bits, string win_type, bool is_dct, bool add_rand_noise) 
:num_mfcc_features(num_mfcc_features), 
 frame_len(frame_len), 
 mfcc_dec_bits(mfcc_dec_bits),
 _is_dct(is_dct),
 _add_rand_noise(add_rand_noise),
 _win_type(win_type)
{

  // Round-up to nearest power of 2.
  frame_len_padded = pow(2,ceil((log(frame_len)/log(2))));
  frame = new float[frame_len_padded];
  energy = new float[frame_len_padded/2+1];
  mel_energies = new float[NUM_FBANK_BINS];
  window_func = _create_window(frame_len);

  //create mel filterbank
  fbank_filter_first = new int32_t[NUM_FBANK_BINS];
  fbank_filter_last = new int32_t[NUM_FBANK_BINS];;
  mel_fbank = create_mel_fbank();
  
  //create DCT matrix
  if(_is_dct) dct_matrix = create_dct_matrix(NUM_FBANK_BINS, num_mfcc_features);
  if(_add_rand_noise) {
    rand_noise = _create_rand_noise(frame_len);
  }

  kiss_fftr_state_ = kiss_fftr_alloc(frame_len_padded, 0, 0, 0);
  cframe_ = new kiss_fft_cpx[frame_len_padded];
}

MFCC::~MFCC() {
  if(frame != nullptr) delete [] frame;
  if(energy != nullptr) delete [] energy;
  if(mel_energies != nullptr) delete [] mel_energies;
  if(window_func != nullptr) delete [] window_func;
  if(fbank_filter_first != nullptr) delete [] fbank_filter_first;
  if(fbank_filter_last != nullptr) delete [] fbank_filter_last;
  if(dct_matrix != nullptr) delete [] dct_matrix;

  for(int i=0;i<NUM_FBANK_BINS;i++)
    if(mel_fbank[i] != nullptr) delete mel_fbank[i];
  if(mel_fbank != nullptr) delete mel_fbank;

  if(_add_rand_noise && rand_noise != nullptr) delete rand_noise;
  if(cframe_ != nullptr) delete cframe_;
  kiss_fftr_free(kiss_fftr_state_);
}

float * MFCC::_create_rand_noise(int len){
  srand((unsigned)time(NULL));
  float * randNoise = new float[len];
  for(int i = 0 ; i < len; ++i){
    randNoise[i] = _randGauss();
  }
  return randNoise;
}

float * MFCC::create_dct_matrix(int32_t input_length, int32_t coefficient_count) {
  int32_t k, n;
  float * M = new float[input_length*coefficient_count];
  float normalizer = sqrtf(2.0/(float)input_length);
  
  for (k = 0; k < coefficient_count; k++) {
    for (n = 0; n < input_length; n++) {
      M[k*input_length+n] = normalizer * cos( ((double)M_PI)/input_length * (n + 0.5) * k );
    }
  }
  return M;
}

float ** MFCC::create_mel_fbank() {

  int32_t bin, i;

  int32_t num_fft_bins = frame_len_padded/2;
  float fft_bin_width = ((float)SAMP_FREQ) / frame_len_padded;
  float mel_low_freq = MelScale(MEL_LOW_FREQ);
  float mel_high_freq = MelScale(MEL_HIGH_FREQ); 
  float mel_freq_delta = (mel_high_freq - mel_low_freq) / (NUM_FBANK_BINS+1);

  float *this_bin = new float[num_fft_bins];

  float ** mel_fbank =  new float*[NUM_FBANK_BINS];

  for (bin = 0; bin < NUM_FBANK_BINS; bin++) {

    float left_mel = mel_low_freq + bin * mel_freq_delta;
    float center_mel = mel_low_freq + (bin + 1) * mel_freq_delta;
    float right_mel = mel_low_freq + (bin + 2) * mel_freq_delta;

    int32_t first_index = -1, last_index = -1;

    for (i = 0; i < num_fft_bins; i++) {

      float freq = (fft_bin_width * i);  // center freq of this fft bin.
      float mel = MelScale(freq);
      this_bin[i] = 0.0;

      if (mel > left_mel && mel < right_mel) {
        float weight;
        if (mel <= center_mel) {
          weight = (mel - left_mel) / (center_mel - left_mel);
        } else {
          weight = (right_mel-mel) / (right_mel-center_mel);
        }
        this_bin[i] = weight;
        if (first_index == -1)
          first_index = i;
        last_index = i;
      }
    }

    fbank_filter_first[bin] = first_index;
    fbank_filter_last[bin] = last_index;
    mel_fbank[bin] = new float[last_index-first_index+1]; 

    int32_t j = 0;
    //copy the part we care about
    for (i = first_index; i <= last_index; i++) {
      mel_fbank[bin][j++] = this_bin[i];
    }
  }
  delete []this_bin;
  return mel_fbank;
}

float * MFCC::_create_window(int frame_length){
  double a = M_2PI / (frame_length-1);
  float * window = new float[frame_length];
  for (int i = 0; i < frame_length; i++) {
    double i_fl = static_cast<double>(i);
    if (_win_type == "hanning") {
      window[i] = 0.5  - 0.5*cos(a * i_fl);
    } else if (_win_type == "sine") {
      // when you are checking ws wikipedia, please
      // note that 0.5 * a = M_PI/(frame_length-1)
      window[i] = sin(0.5 * a * i_fl);
    } else if (_win_type == "hamming") {
      window[i] = 0.54 - 0.46*cos(a * i_fl);
    } else if (_win_type == "povey") {  
      window[i] = pow(0.5 - 0.5*cos(a * i), 0.85);
    } else if (_win_type == "rectangular") {
      window[i] = 1.0;
    } else if (_win_type == "blackman") {
      float blackman_coeff = 0.42;
      window[i] = blackman_coeff - 0.5*cos(a * i_fl) +
        (0.5 - blackman_coeff) * cos(2 * a * i_fl);
    } else {
      printf("Invalid window type %s\n", _win_type);
      window[i] = 1.0;
    }
  }
  return window;
}

void MFCC::_preEmphasize(float *frame, int len, float preemph_coeff) {
  if (preemph_coeff == 0.0) return;
  for (int i = len-1; i > 0; i--)
    frame[i] -= preemph_coeff * frame[i-1];
  frame[0] -= preemph_coeff * frame[0];
}


void MFCC::mfcc_compute(const int16_t * audio_data, float* mfcc_out)
{
  int i;
  for (i = 0; i < frame_len; i++) {
    frame[i] = (float)audio_data[i]/(1<<15); 
  }  
  memset(&frame[frame_len], 0, sizeof(float) * (frame_len_padded-frame_len));
  mfcc_compute_scaled(mfcc_out);
}

void MFCC::mfcc_compute(const float * audio_data, float* mfcc_out) {  
  int i;
   for (i = 0; i < frame_len; i++) {
    frame[i] = audio_data[i]/(1<<15); 
  }
  memset(&frame[frame_len], 0, sizeof(float) * (frame_len_padded-frame_len));
  mfcc_compute_scaled(mfcc_out);
}

void MFCC::mfcc_compute_scaled(float * mfcc_out)
{
  this->fbank_compute_scaled(mel_energies);
  //Take DCT. Uses matrix mul.
  for (int32_t i = 0; i < num_mfcc_features; i++) {
    float sum = 0.0;
    for (int32_t j = 0; j < NUM_FBANK_BINS; j++) {
      sum += dct_matrix[i*NUM_FBANK_BINS+j] * mel_energies[j];
    }

    mfcc_out[i] = sum;  
  }
}

void MFCC::fbank_compute_scaled(float * fbank_out)
{
  static const float ENERGY_MIN = 1e-12;
  int32_t i, j, bin;
  for (i = 0; i < frame_len; i++) {
    frame[i] *= window_func[i];
  } 

  //Convert to power spectrum, FFT
  kiss_fftr(kiss_fftr_state_, frame, cframe_);
  for(int i=0; i<frame_len_padded/2+1;i++){
      energy[i] = cframe_[i].i*cframe_[i].i+cframe_[i].r*cframe_[i].r;
  }
 
  float sqrt_data;
  //Apply mel filterbanks
  for (bin = 0; bin < NUM_FBANK_BINS; bin++) {
    j = 0;
    float mel_energy = 0;
    int32_t first_index = fbank_filter_first[bin];
    int32_t last_index = fbank_filter_last[bin];
    for (i = first_index; i <= last_index; i++) {
      sqrt_data = sqrtf(energy[i]);
      mel_energy += (sqrt_data) * mel_fbank[bin][j++];
    }
    mel_energies[bin] = mel_energy;

    if (mel_energy < ENERGY_MIN)
        mel_energies[bin] = ENERGY_MIN;
  }

  //Take log
  for (bin = 0; bin < NUM_FBANK_BINS; bin++)
    mel_energies[bin] = logf(mel_energies[bin]);
}

void MFCC::fbank_compute_kaldilike(float * fbank_out)
{
  static const float ENERGY_MIN = 1e-12;
  int32_t i, j, bin;

  //去直流分量
  bool remove_dc_offset = true;
  if (remove_dc_offset){
    float sum = 0.0;
    for(int i = 0; i < frame_len; ++i){
        sum += frame[i];
    }
    sum /= frame_len;
    for(int i = 0; i < frame_len; ++i){
        frame[i] -= sum;
    }
  }

  //预加重
  _preEmphasize(frame, frame_len, 0.97);

  //加窗
  for (i = 0; i < frame_len; i++) {
    frame[i] *= window_func[i];
  }

  //FFT, 功率谱计算，kaldi没有乘1.0/nfft
  kiss_fftr(kiss_fftr_state_, frame, cframe_);
  for(int i=0; i<frame_len_padded/2+1;i++){
      energy[i] = cframe_[i].r*cframe_[i].r + cframe_[i].i*cframe_[i].i;
  }
 
  //梅尔滤波
  for (int bin = 0; bin < NUM_FBANK_BINS; bin++) {
    float mel_energy = 0;
    int32_t first_index = fbank_filter_first[bin];
    int32_t last_index = fbank_filter_last[bin];
    j = 0;
    for (int i = first_index; i <= last_index; i++) {
      mel_energy += energy[i] * mel_fbank[bin][j++];
    }
    fbank_out[bin] = mel_energy;

    if (mel_energy < ENERGY_MIN)
        fbank_out[bin] = ENERGY_MIN;
  }

  //Take log
  for (bin = 0; bin < NUM_FBANK_BINS; bin++)
    fbank_out[bin] = logf(fbank_out[bin]);
}
ofstream foutres("mfcc.fbank.txt");
void MFCC::fbank_compute(const float * audio_data, bool kaldi_like, float* fbank_out) {  
  if(kaldi_like){
    //+ 加随机扰动
    for (int i = 0; i < frame_len; i++) {
      if(_add_rand_noise) frame[i] = audio_data[i] + rand_noise[i]; 
      else frame[i] = audio_data[i];
    }
    memset(&frame[frame_len], 0, sizeof(float) * (frame_len_padded-frame_len));
    fbank_compute_kaldilike(fbank_out);
    for(int s = 0 ; s < 40; s++){
        foutres << fbank_out[s] << "\n";
    }
  }else{
    for (int i = 0; i < frame_len; i++) {
      frame[i] = audio_data[i]/(1<<15); 
    }
    memset(&frame[frame_len], 0, sizeof(float) * (frame_len_padded-frame_len));
    fbank_compute_scaled(fbank_out);
  }
}

}



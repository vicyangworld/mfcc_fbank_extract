from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import math
import os

M_2PI = 6.283185307179586476925286766559005

sample_rate, wav = wavfile.read('1.wav')
wav = np.array(wav)

allsamples = wav.size
print('all sample: ', allsamples, 'sample rate: ', sample_rate)
plt.plot(wav, c='red')
plt.show()

frame_length_ms = 25
frame_shift_ms  = 10
add_rand_noise = True

# step 1: 分帧 + 加随机扰动（训练）
num_of_frames = (allsamples*1000//sample_rate - frame_length_ms) // frame_shift_ms
num_of_per_frame = (sample_rate * frame_length_ms // 1000)
assert(num_of_per_frame == 400)
print('all frames: ', num_of_frames, 'num_of_per_frame', num_of_per_frame)
frames = np.zeros((num_of_frames, num_of_per_frame))
for f in range(num_of_frames):
    idx = f * (frame_shift_ms * sample_rate // 1000)
    frames[f,:]= wav[idx:idx+num_of_per_frame]
    if add_rand_noise:
        def gauss_rand(n):
            u = np.random.uniform(0,1,n)
            r = np.sqrt(-2.0 * np.log(u)) * np.cos(M_2PI*u)
            return r
        rand_arr = gauss_rand(num_of_per_frame)
        # frames[f] = frames[f] + rand_arr
        frames[f] = frames[f]/(1<<15) + rand_arr
plt.plot(frames[1], c='green')
plt.show()

# step 2: 去直流分量
for f in range(num_of_frames):
    mean = np.mean(frames[f,:])
    frames[f,:] = frames[f,:] - mean
plt.plot(frames[1], c='black')
plt.show()

# step 3: 预加重
preemph_coeff = 0.97
for f in range(num_of_frames):
    frames[f][1:] = frames[f][1:] - preemph_coeff * frames[f][:-1]
    frames[f][0] = frames[f][0] - preemph_coeff * frames[f][0]
plt.plot(frames[1], c='pink')
plt.show()

# step 4: 加窗，povey窗
a = M_2PI / (num_of_per_frame-1)
window = np.array([math.pow(0.5 - 0.5*math.cos(a * float(i)), 0.85) for i in range(num_of_per_frame)])
for f in range(num_of_frames):
    frames[f] = frames[f] * window
plt.plot(frames[1], c='blue')
plt.show()

# step 5: Padding
ntemp = num_of_per_frame
cnt = 0
while(ntemp):
    ntemp = ntemp>>1
    cnt = cnt + 1
num_of_per_frame_padding = int(math.pow(2, cnt))
padding_array = np.zeros(num_of_per_frame_padding-num_of_per_frame)
frames_padding = np.zeros((num_of_frames, num_of_per_frame_padding))
for f in range(num_of_frames):
    frames_padding[f] = np.append(frames[f], padding_array)
plt.plot(frames_padding[1], c='blue')
plt.show()
print('frames_padding.shape: ', frames_padding.shape)

# step 6:  FFT, 功率谱计算，kaldi没有乘1.0/nfft
nfft = num_of_per_frame_padding
half_dim = nfft // 2
energy_frames = np.zeros((num_of_frames,half_dim))
for f in range(num_of_frames):
    energy_frames[f] = np.absolute(np.fft.rfft(frames_padding[f], nfft))[:-1] ** 2
plt.plot(energy_frames[1], c='gray')
plt.show()

# step 7: 构建梅尔滤波器
num_of_bins = 40
low_freq = 100.0
high_freq = 7800.0

fft_bin_width = 1.0* sample_rate / nfft
mel_low_freq  = 1127.0 * np.log(1.0 + low_freq / 700.0)
mel_high_freq = 1127.0 * np.log(1.0 + high_freq / 700.0)
mel_freq_delta = float((mel_high_freq - mel_low_freq) / (num_of_bins + 1))
mel_coef = np.zeros((num_of_bins,half_dim+2))   # nfft/2 + 1 + 1，后面两个1为首尾频率点索引，第几个频率点开始，第几个频率点结束
for b in range(num_of_bins):
    left_mel = mel_low_freq + b* mel_freq_delta
    center_mel = mel_low_freq + (b+1) * mel_freq_delta
    right_mel = mel_low_freq + (b+2) * mel_freq_delta

    first_index = -1
    last_index = -1
    for i in range(half_dim):
        freq = fft_bin_width * float(i)
        mel = 1127.0 * np.log(1.0 + freq / 700.0)

        if mel > left_mel and mel < right_mel:
            weight = 0
            if mel <= center_mel:
                weight = (mel - left_mel) / (center_mel - left_mel)
            else:
                weight = (right_mel - mel) / (right_mel - center_mel)
            mel_coef[b][i] = weight

            if first_index == -1:
                first_index = i
            last_index = i

    mel_coef[b][half_dim] = first_index     #为了减少计算量，避免与零相乘， 这里没有用到，c++里可以用上
    mel_coef[b][half_dim+1] = last_index    #为了减少计算量，避免与零相乘， 这里没有用到，c++里可以用上

for f in range(num_of_bins):
    plt.plot(mel_coef[f,:half_dim])
plt.show()

# step 8: 梅尔滤波
mel_frames = np.zeros((num_of_frames, num_of_bins))
for f in range(num_of_frames):
    for b in range(num_of_bins):
        mel_frames[f][b] = np.sum(energy_frames[f,:half_dim] * mel_coef[b,:half_dim])  #可以用矩阵相乘

# step 9: 取对数
log_mel_frames = np.log(mel_frames)

np.savetxt('fbank_fea.txt', log_mel_frames.reshape(-1), fmt='%.10f')
import numpy as np
import torch
import torchaudio
import librosa
import os
from glob import glob

num_mels=80
n_fft=1024
hop_size=256
win_size=1024
sampling_rate=22050
fmin=0
fmax=8000

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    device = y.device
    melTorch = torchaudio.transforms.MelSpectrogram(sample_rate=sampling_rate, n_fft=n_fft, n_mels=num_mels, \
           hop_length=hop_size, win_length=win_size, f_min=fmin, f_max=fmax, pad=int((n_fft-hop_size)/2), center=center).to(device)      
    spec = melTorch(y)
    return spec

def to_mono(audio, dim=-2): 
    if len(audio.size()) > 1:
        return torch.mean(audio, dim=dim, keepdim=True)
    else:
        return audio

def load_audio(audio_path, sr=None, mono=True):
    if 'mp3' in audio_path:
        torchaudio.set_audio_backend('sox_io')
    audio, org_sr = torchaudio.load(audio_path)
    audio = to_mono(audio) if mono else audio
    
    if sr and org_sr != sr:
        audio = torchaudio.transforms.Resample(org_sr, sr)(audio)

    return audio

if __name__ == '__main__':
    stft = torchaudio.transforms.InverseMelScale(n_stft=n_fft//2 + 1, n_mels=num_mels, sample_rate=sampling_rate, f_min=fmin, f_max=fmax).cuda()
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=n_fft, n_iter=1024, win_length=win_size, hop_length=hop_size).cuda()
    
    load_audio_path = '/home/yuehpo/coding/DeepMIR_2023fall/hw2/data/m4singer_22.5k_symbolic'
    save_npy_path = '/home/yuehpo/coding/hifi-gan/ft_dataset'
    if not os.path.exists(save_npy_path):
        os.mkdir(save_npy_path)
    # audio_list = os.listdir(load_audio_path)
    audio_list = glob(os.path.join(load_audio_path, "**", '*.wav'), recursive=True)
    audio_list.sort()
    for audio in audio_list:
        print(f"Processing {audio}...")
        y = load_audio(os.path.join(load_audio_path, audio), sr=sampling_rate)
        mel_tensor = mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax)
        mel = mel_tensor.squeeze().cpu().numpy()
        # file_name = os.path.join(save_npy_path, "-".join(audio.split('/')[-2:]).replace('.wav', '.npy'))
        file_name = os.path.join(save_npy_path, audio.split('/')[-1].replace('.wav', '.npy'))
        print(f"Saving {file_name}...")
        np.save(file_name, mel)
        mel = np.load(file_name) # check the .npy is readable
        
        # # transform mel to audio and save it
        # mel_tensor = torch.from_numpy(mel).unsqueeze(0) if len(mel.shape) < 3 else torch.from_numpy(mel)
        # wavform = griffin_lim(stft(mel_tensor.cuda()))
        # # save audio
        # torchaudio.save(os.path.join(save_npy_path, audio.split('/')[-1]), wavform.cpu(), sampling_rate)
        
        
        

    # plot the last melspectrogram
    # ref: https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # # don't forget to do dB conversion
    # S_dB = librosa.power_to_db(S, ref=np.max)
    # img = librosa.display.specshow(S_dB, x_axis='time',
    #                          y_axis='mel', sr=sampling_rate,
    #                          fmax=fmax, ax=ax, hop_length=hop_size, n_fft=n_fft)
    # fig.colorbar(img, ax=ax, format='%+2.0f dB')
    # ax.set(title='Mel-frequency spectrogram')
    

import glob
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import gc

plt.style.use('ggplot')
# plt.rcParams['font.family'] = 'DejaVu Sans'
# plt.rcParams['font.serif'] = 'Ubuntu'
# plt.rcParams['font.monospace'] = 'Ubuntu Mono'
# plt.rcParams['font.size'] = 12
# plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['axes.titlesize'] = 14
# plt.rcParams['xtick.labelsize'] = 10
# plt.rcParams['ytick.labelsize'] = 10
# plt.rcParams['legend.fontsize'] = 11
# plt.rcParams['figure.titlesize'] = 13

def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds

def plot_waves(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(10, 12), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(5,1,i)
        plt.tight_layout()
        librosa.display.waveplot(np.array(f),sr=22050)
        # plt.colorbar(format='%+2.0f dB')
        plt.title(n.title())
        i += 1
        # plt.clf()
    # plt.suptitle("Figure 1: Waveplot",x=0.5, y=0.915,fontsize=18)
    plt.savefig('Waveplot.png')
    fig.clf()    # normally I use these lines to release the memory
    plt.close()
   
    gc.collect()
    # plt.close(fig)
    # fig.close()
    # plt.show()

def plot_librosa_specgram(sound_names,raw_sounds):
    i = 1
    # fig = plt.figure(figsize=(25,60), dpi = 900)
    fig = plt.figure(figsize=(10, 15), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(5,1,i)
        D = librosa.amplitude_to_db(librosa.stft(f), ref=np.max)
        librosa.display.specshow(D, y_axis='linear') 
        plt.colorbar(format='%+2.0f dB')
        plt.title(n.title())
        i += 1
    #plt.suptitle("Figure 2: Linear-frequency power spectrogram",x=0.5, y=0.915,fontsize=18)
    plt.tight_layout()
    plt.savefig('Spectrogram.png')
    plt.clf()
    
def plot_librosa_logspecgram(sound_names,raw_sounds):
    i = 1
    # fig = plt.figure(figsize=(25,60), dpi = 900)
    fig = plt.figure(figsize=(10, 15), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(5,1,i)
        D = librosa.amplitude_to_db(librosa.stft(f), ref=np.max)
        librosa.display.specshow(D, y_axis='log') 
        plt.colorbar(format='%+2.0f dB')
        plt.title(n.title())
        i += 1
    #plt.suptitle("Figure 2: Log-frequency power spectrogram",x=0.5, y=0.915,fontsize=18)
    plt.tight_layout()
    plt.savefig('LogSpectrogram.png')
    plt.clf()
    

def plot_specgram(sound_names,raw_sounds):
    i = 1
    # fig = plt.figure(figsize=(25,60), dpi = 900)
    fig = plt.figure(figsize=(10, 15), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(5,1,i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    #plt.suptitle("Figure 2: Spectrogram",x=0.5, y=0.915,fontsize=18)
    plt.tight_layout()
    plt.savefig('Spectrogram-mmm.png')
    plt.clf()
    # plt.show()

def plot_log_power_specgram(sound_names,raw_sounds):
    i = 1
    # fig = plt.figure(figsize=(25,60), dpi = 900)
    fig = plt.figure(figsize=(10, 8), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        D = librosa.logamplitude(np.abs(librosa.stft(f))**2, ref_power=np.max)
        librosa.display.specshow(D,x_axis='time' ,y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 3: Log power spectrogram",x=0.5, y=0.915,fontsize=18)
    # plt.show()
    plt.tight_layout()
    plt.savefig('Log power spectrogram.png')
    plt.clf()


def plot_mel_specgram(sound_names,raw_sounds):
    i = 1
    # fig = plt.figure(figsize=(25,60), dpi = 900)
    fig = plt.figure(figsize=(10, 15), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(5,1,i)
        S = librosa.feature.melspectrogram(y=f, n_mels=128,
                                    fmax=8000)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                            y_axis='mel', fmax=8000,
                            x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title(n.title())
        i += 1
    #plt.suptitle("Figure 2: Spectrogram",x=0.5, y=0.915,fontsize=18)
    plt.tight_layout()
    plt.savefig('Mellll-mmm.png')
    plt.clf()
    # plt.show()

def plot_mfcc(sound_names,raw_sounds):
    i = 1
    # fig = plt.figure(figsize=(25,60), dpi = 900)
    fig = plt.figure(figsize=(10, 15), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(5,1,i)
        mfccs = librosa.feature.mfcc(y=f) #, n_mfcc=40
        librosa.display.specshow(mfccs, x_axis='time')  
        plt.colorbar()
        plt.title(n.title())
        i += 1
    #plt.suptitle("Figure 2: Spectrogram",x=0.5, y=0.915,fontsize=18)
    plt.tight_layout()
    plt.savefig('MFCC-mmm.png')
    plt.clf()
    # plt.show()    
    
def plot_cromogram(sound_names,raw_sounds):
    i = 1
    # fig = plt.figure(figsize=(25,60), dpi = 900)
    fig = plt.figure(figsize=(10, 15), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(5,1,i)
        stft = np.abs(librosa.stft(f))
        chroma = librosa.feature.chroma_stft(S=stft)
        librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')  
        plt.colorbar()
        plt.title(n.title())
        i += 1
    #plt.suptitle("Figure 2: Spectrogram",x=0.5, y=0.915,fontsize=18)
    plt.tight_layout()
    plt.savefig('cromaaaa-mmm.png')
    plt.clf()
    # plt.show()   

def plot_tonnetz(sound_names,raw_sounds):
    i = 1
    # fig = plt.figure(figsize=(25,60), dpi = 900)
    fig = plt.figure(figsize=(10, 15), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(5,1,i)
        #stft = np.abs(librosa.stft(f))
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(f))
        librosa.display.specshow(tonnetz, y_axis='tonnetz')
        plt.colorbar()
        plt.title(n.title())
        i += 1
    #plt.suptitle("Figure 2: Spectrogram",x=0.5, y=0.915,fontsize=18)
    plt.tight_layout()
    plt.savefig('tonnetz-mmm.png')
    plt.clf()
    # plt.show()       


sound_file_paths = ["data/data_for_plots/normal__153_1306848820671_B.wav", 
    "data/data_for_plots/murmur__156_1306936373241_B.wav", 
    "data/data_for_plots/extrastole__213_1308245263936_D.wav", "data/data_for_plots/artifact__201012172012.wav",
    "data/data_for_plots/extrahls__201101160808.wav"]

sound_names = ["Normal", "Murmur", "Extrastole", "Artifact", "Extrahls"]

raw_sounds = load_sound_files(sound_file_paths)

plot_tonnetz(sound_names,raw_sounds)
# plot_waves(sound_names,raw_sounds)
# plot_librosa_specgram(sound_names,raw_sounds)
# plot_librosa_logspecgram(sound_names,raw_sounds)
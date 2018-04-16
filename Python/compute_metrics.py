# Beat tracking example
from __future__ import print_function
import os
import glob
import ntpath
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import csv
# 1. Get the file path to the included audio example
#filename = librosa.util.example_audio_file()
#labels_map = ['artifact', 'normal', 'murmur', 'extrahls'] #setb labels
# labels_map = ['normal', 'murmur', 'extrastole'] #setb labels
labels_map = ['artifact', 'extrahls', 'normal', 'murmur', 'extrastole', 'Aunlabelledtest', 'Bunlabelledtest'] #all
# labels_map = ['artifact', 'normal', 'murmur', 'extrastole']#{'artifact':0, 'murmur':1}
#filename = "data/set_a/murmur__201108222221.wav"

data = {}
# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`
# y, sr = librosa.load(filename)
# # 3. Run the default beat tracker
# tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

# #data['chroma_stft'] = librosa.feature.chroma_stft(y=y, sr=sr)
# data['beat_track'] = librosa.frames_to_time(beat_frames, sr=sr)
# beat_times = librosa.frames_to_time(beat_frames, sr=sr)
# print(beat_times)


def extract_feature(file_name):
	X, sample_rate = librosa.load(file_name)
	stft = np.abs(librosa.stft(X))
	mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
	chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
	mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
	contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
	tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)

	# Compute local onset autocorrelation
	
	#hop_length = 512
	oenv = librosa.onset.onset_strength(y=X, sr=sample_rate)
	tempogram = np.mean(librosa.feature.tempogram(onset_envelope=oenv, sr=sample_rate).T, axis=0)
	# Compute global onset autocorrelation
	# ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
	# ac_global = librosa.util.normalize(ac_global)

	# rmse = np.mean(librosa.feature.rmse(y=X).T, axis=0)
	# cent = np.mean(librosa.feature.spectral_centroid(y=X, sr=sample_rate).T, axis=0)
	# spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=X, sr=sample_rate).T, axis=0)

	#print("tempogram:%d, ac_global:%d, rmse:%d, stft:%d, mel:%d" % (len(tempogram), len(ac_global), len(rmse), len(stft), len(mel)))
	# print(rmse)
	# print(cent)
	# print(spec_bw)

	# onset_env = librosa.onset.onset_strength(y=X, sr=sample_rate,
	#                                          hop_length=512,
	#                                          aggregate=np.median)
	# peaks = librosa.util.peak_pick(onset_env, 3, 3, 3, 5, 0.5, 10)
	#print(stft)
	#print(tonnetz)
	#print(peaks)
	return mfccs,chroma,mel,contrast,tonnetz,tempogram

def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
	# print(enumerate(sub_dirs))

	# filenames, folders = []
	saveft = []
	header = []
	features, labels = np.empty((0,580)), np.empty(0)
	for sub_dir in sub_dirs:#enumerate(sub_dirs):
		for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
			file_name = ntpath.basename(fn)
			file_attrs = file_name.split("_")
			xi_class = file_attrs[0].strip()
			            
			# if xi_class != "Bunlabelledtest":
			#print('Processing file: %s'.ljust(30) % file_name, end='\r')
			mfccs, chroma, mel, contrast, tonnetz, tempogram = extract_feature(fn)
			# print(mfccs.shape)
			# print(tonnetz.shape)
			# print(tempogram.shape)
			# print(ac_global.shape)

			ext_features = np.hstack([file_name,sub_dir,xi_class,mfccs,chroma,mel,contrast,tonnetz,tempogram])

			header = ['filename', 'folder', 'label']
			header.extend(['mfcc']*len(mfccs))
			header.extend(['chroma']*len(chroma))
			header.extend(['mel']*len(mel))
			header.extend(['contrast']*len(contrast))
			header.extend(['tonnetz']*len(tonnetz))
			header.extend(['tempogram']*len(tempogram))
			# header.extend(['ac_global']*len(ac_global))

			tmp = [file_name, sub_dir, xi_class]
			tmp.extend(mfccs)
			tmp.extend(chroma)
			tmp.extend(mel )
			tmp.extend(contrast)
			tmp.extend(tonnetz)
			tmp.extend(tempogram)
			# tmp.extend(ac_global)
			#print(len(ext_features))
			features = np.vstack([features,ext_features])
			#print(labels_map[file_attrs[0].strip()])
			labels = np.append(labels, labels_map.index(xi_class))
			# filenames.append(file_name)
			# folders.append(sub_dir)
			saveft.append(tmp)
			# exit(-1)

	# print(mfccs.tolist())
	
	# with open('some.csv', 'w') as f:
	#     writer = csv.writer(f)
	#     writer.writerow(mfccs.tolist())
	#     writer.writerow(chroma.tolist())
	#     writer.writerow(mel.tolist())
	saveft.insert(0, header)
	return np.array(features), np.array(labels, dtype = np.int), saveft

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


parent_dir = 'data'
sub_dirs = ['set_a', 'set_b']
features, labels, saveft = parse_audio_files(parent_dir, sub_dirs)
# print(labels)
# labels = one_hot_encode(labels)

# np.savetxt("features.csv", features, delimiter=",")
# np.savetxt("labels.csv", labels, delimiter=",")

print(len(saveft))
import csv
with open('features.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerows(saveft)


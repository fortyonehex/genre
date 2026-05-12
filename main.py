import librosa
import numpy as np
import matplotlib.pyplot as plt
import model
import config

# -----------------------------------
# THE MOST IMPORTANT VARIABLES YOU SHOULD CHANGE 
# the length (in seconds) of each input you want to pass to the model
input_length = 3

# the time skipped (in seconds) between each input you want to pass to the model
overlap_length = 3

# the path to the audio file you wish to predict
audio_file = './test_audio/TRWJAZW128F42760DD_test.mp3'
# ------------------------------------

labels = config.MSD_LABELS
n_classes = config.MSD_LENGTH
n_frames = librosa.time_to_frames(input_length, sr=config.SR, n_fft=config.FFT_SIZE, hop_length=config.FFT_HOP) + 1
overlap = librosa.time_to_frames(overlap_length, sr=config.SR, n_fft=config.FFT_SIZE, hop_length=config.FFT_HOP) + 1

# computing log-mel spectrogram
print("Loading audio")
audio, sr = librosa.load(audio_file, sr=config.SR)
audio_rep = librosa.feature.melspectrogram(y=audio, sr=sr, 
                                           hop_length=config.FFT_HOP,
                                           n_fft=config.FFT_SIZE,
                                           n_mels=config.N_MELS).T
audio_rep = audio_rep.astype(np.float32)
audio_rep = np.log10(10000 * audio_rep + 1)

# batch the audio
print("Batching audio")
first = True
last_frame = audio_rep.shape[0] - n_frames + 1
for time_stamp in range(0, last_frame, overlap):
    patch = np.expand_dims(audio_rep[time_stamp:time_stamp+n_frames, :], axis=0)
    if first:
        batch = patch
        first = False
    else:
        batch = np.concatenate((batch, patch), axis = 0)

# load the model itself
print("Loading model")
pred_model = model.load_model(config.PRETRAINED_LOC, n_frames)

print("Predicting")
predicted_tags = np.exp(pred_model.predict(batch))

sum_probs = np.sum(predicted_tags, axis=0)
top_labels = [labels[x] for x in np.argsort(sum_probs)[::-1]]
print("Top 5 labels:", ", ".join(top_labels[:5]))

fig, ax = plt.subplots()
im = ax.imshow(predicted_tags.T, interpolation='nearest', aspect='auto')
ax.set_yticks(range(n_classes), labels=labels)
plt.show()
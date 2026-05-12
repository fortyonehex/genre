from musicnn.musicnn import configuration as config
import tensorflow as tf
from tensorflow import keras
from keras import layers, activations
import os

keras.utils.set_random_seed(42)

labels = config.MSD_LABELS
n_classes = len(labels)

layer_seq = {}
def layer_name(layer_type):
    global layer_seq
    if layer_type in layer_seq:
        layer_seq[layer_type] += 1
        return f'{layer_type}_{layer_seq[layer_type]}'
    else:
        layer_seq[layer_type] = 0
        return layer_type

# changes depending on model -- remove hardcoding later
front_n_filters = 1.6
mid_n_filters = 512
back_n_units = 500

def timbral_block(x, kernel_height):
    x = layers.Conv2D(filters=int(front_n_filters*128),
                      kernel_size=[7, kernel_height],
                      padding='valid',
                      activation=activations.relu,
                      name=layer_name('conv2d'))(x)
    x = layers.BatchNormalization(name=layer_name('batch_normalization'))(x)
    x = layers.MaxPooling2D(pool_size=[1, x.shape[2]],
                            strides=[1, x.shape[2]])(x)
    x = x.squeeze(axis=2)
    return x

def tempo_block(x, kernel_width):
    x = layers.Conv2D(filters=int(front_n_filters*32),
                      kernel_size=[kernel_width, 1],
                      padding='same',
                      activation=activations.relu,
                      name=layer_name('conv2d'))(x)
    x = layers.BatchNormalization(name=layer_name('batch_normalization'))(x)
    x = layers.MaxPooling2D(pool_size=[1, x.shape[2]],
                            strides=[1, x.shape[2]])(x)
    x = x.squeeze(axis=2)
    return x

def resnet_layer(x):
    x = keras.ops.pad(x, [[0, 0], [3, 3], [0, 0], [0, 0]], 'constant')
    x = layers.Conv2D(filters=mid_n_filters,
                      kernel_size=[7, x.shape[2]],
                      padding='valid',
                      activation=activations.relu,
                      name=layer_name('conv2d'))(x)
    x = layers.BatchNormalization(name=layer_name('batch_normalization'))(x)
    x = keras.ops.transpose(x, [0, 1, 3, 2])
    return x

# ------------
# for testing purposes

import librosa
import numpy as np

audio_file = 'C:\\Users\\Bobby\\Documents\\Coding\\genre\\musicnn\\audio\\TRWJAZW128F42760DD_test.mp3'

print("Loading audio...")

input_length = 3
n_frames = librosa.time_to_frames(input_length, sr=config.SR, n_fft=config.FFT_SIZE, hop_length=config.FFT_HOP) + 1
overlap = n_frames

# computing log-mel spectrogram
audio, sr = librosa.load(audio_file, sr=config.SR)
audio_rep = librosa.feature.melspectrogram(y=audio, sr=sr, 
                                           hop_length=config.FFT_HOP,
                                           n_fft=config.FFT_SIZE,
                                           n_mels=config.N_MELS).T
audio_rep = audio_rep.astype(np.float32)
audio_rep = np.log10(10000 * audio_rep + 1)

# batch
first = True
last_frame = audio_rep.shape[0] - n_frames + 1
for time_stamp in range(0, last_frame, overlap):
    patch = np.expand_dims(audio_rep[time_stamp:time_stamp+n_frames, :], axis=0)
    if first:
        batch = patch
        first = False
    else:
        batch = np.concatenate((batch, patch), axis = 0)

print("Loaded and batched audio.")
# ------------

inputs = keras.Input(shape=(n_frames, config.N_MELS))
inputs = keras.ops.expand_dims(inputs, 3)
inputs_norm = layers.BatchNormalization(name=layer_name('batch_normalization'))(inputs)
inputs_padded = keras.ops.pad(inputs_norm, [[0, 0], [3, 3], [0, 0], [0, 0]], 
                              'constant')

timbral_1 = timbral_block(inputs_padded, int(0.4 * config.N_MELS))
timbral_2 = timbral_block(inputs_padded, int(0.7 * config.N_MELS))
tempo_1 = tempo_block(inputs_norm, 128)
tempo_2 = tempo_block(inputs_norm, 64)
tempo_3 = tempo_block(inputs_norm, 32)

comb_features = layers.Concatenate(axis=2)([timbral_1, timbral_2, 
                                            tempo_1, tempo_2, tempo_3])
comb_features = keras.ops.expand_dims(comb_features, 3)

conv1 = resnet_layer(comb_features)
conv2 = resnet_layer(conv1)
res2 = layers.Add()(inputs=[conv1, conv2])
conv3 = resnet_layer(res2)
res3 = layers.Add()(inputs=[res2, conv3])
comb_res = layers.Concatenate(axis=2)([comb_features, conv1, res2, res3])

max_pool = keras.ops.max(comb_res, axis=1)
mean_pool = keras.ops.mean(comb_res, axis=1)
tmp_pool = layers.Concatenate(axis=2)([max_pool, mean_pool])

flat_pool = layers.Flatten()(tmp_pool)
flat_pool = layers.BatchNormalization(name=layer_name('batch_normalization'))(flat_pool)
flat_pool = layers.Dropout(rate=0.5)(flat_pool)

dense1 = layers.Dense(units=back_n_units, 
                      activation=activations.relu,
                      name=layer_name('dense'))(flat_pool)
dense1 = layers.BatchNormalization(name=layer_name('batch_normalization'))(dense1)
dense1 = layers.Dropout(rate=0.5)(dense1)

logits = layers.Dense(units=n_classes, 
                      activation=None,
                      name=layer_name('dense'))(dense1)

model = keras.Model(inputs=inputs, outputs=logits)

# hardcoded again -- change if needed
ckpt_path = os.path.dirname(__file__) + '\\musicnn\\musicnn\\MSD_musicnn_big\\'

reader = tf.train.load_checkpoint(ckpt_path)
var_map = reader.get_variable_to_shape_map()
assg_map = {}

# print(var_map)
# print({v.path: v.shape for v in model.variables})

for v in model.variables:
    if v.path in var_map:
        v.assign(tf.train.load_variable(ckpt_path, v.path))
    else:
        print(f"WARNING: {v.path} not in checkpoint")

print(model.variables)

# checkpoint = tf.train.Checkpoint(model)
# status = checkpoint.restore(tf.train.latest_checkpoint(ckpt_path))
# status.assert_existing_objects_matched()
# print(status)

print(repr(batch))
predicted_tags = model(np.expand_dims(batch[0], 0), training=False)

print(repr(predicted_tags))
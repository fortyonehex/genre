import tensorflow as tf
from tensorflow import keras
from keras import layers, activations
import config
import os

layer_seq = {}
def layer_name(layer_type):
    global layer_seq
    if layer_type in layer_seq:
        layer_seq[layer_type] += 1
        return f'{layer_type}_{layer_seq[layer_type]}'
    else:
        layer_seq[layer_type] = 0
        return layer_type

def timbral_block(x, kernel_height):
    x = layers.Conv2D(filters=int(config.FRONT_N_FILTERS*128),
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
    x = layers.Conv2D(filters=int(config.FRONT_N_FILTERS*32),
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
    x = layers.Conv2D(filters=config.MID_N_FILTERS,
                      kernel_size=[7, x.shape[2]],
                      padding='valid',
                      activation=activations.relu,
                      name=layer_name('conv2d'))(x)
    x = layers.BatchNormalization(name=layer_name('batch_normalization'))(x)
    x = keras.ops.transpose(x, [0, 1, 3, 2])
    return x

def build_model(n_frames: int):
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

    dense1 = layers.Dense(units=config.BACK_N_UNITS, 
                        activation=activations.relu,
                        name=layer_name('dense'))(flat_pool)
    dense1 = layers.BatchNormalization(name=layer_name('batch_normalization'))(dense1)
    dense1 = layers.Dropout(rate=0.5)(dense1)

    logits = layers.Dense(units=config.MSD_LENGTH, 
                        activation=None,
                        name=layer_name('dense'))(dense1)

    model = keras.Model(inputs=inputs, outputs=logits)
    return model

def load_model(ckpt_path: str, n_frames: int):
    model = build_model(n_frames)
    abs_ckpt_path = os.path.abspath(ckpt_path)
    if os.name == "nt":
        abs_ckpt_path += "\\"
    else:
        abs_ckpt_path += "/"
    print(abs_ckpt_path)
    
    try:
        reader = tf.train.load_checkpoint(abs_ckpt_path)
    except:
        with open(abs_ckpt_path + 'checkpoint', 'w') as ckpt_if:
            ckpt_if.write(f"model_checkpoint_path: {repr(abs_ckpt_path)}\nall_model_checkpoint_paths: {repr(abs_ckpt_path)}\n")
        reader = tf.train.load_checkpoint(abs_ckpt_path)
    var_map = reader.get_variable_to_shape_map()

    for v in model.variables:
        if v.path in var_map:
            v.assign(tf.train.load_variable(abs_ckpt_path, v.path))
        else:
            print(f"WARNING: {v.path} not in checkpoint")

    return model
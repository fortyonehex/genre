from musicnn.musicnn import configuration as config
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import activations
import tf2onnx
import torch
import os

# automatically increment layer names
layer_seq = {}
def layer_name(layer_type):
    global layer_seq
    if layer_type in layer_seq:
        layer_seq[layer_type] += 1
        return f'{layer_type}_{layer_seq[layer_type]}'
    else:
        layer_seq[layer_type] = 0
        return layer_type

# A port of the models found in musicnn to Tensorflow 2.x.

# def create_model

def timbral_block(inputs, filters, kernel_size, is_training, padding="valid", activation=activations.relu):
    conv = layers.Conv2D(filters=filters, 
                                  kernel_size=kernel_size, 
                                  padding=padding, 
                                  activation=activation,
                                  name=layer_name("conv2d"))(inputs)
    bn_conv = layers.BatchNormalization()(conv, 
                                                   training=is_training)
    pool = layers.MaxPooling2D(pool_size=[1, bn_conv.shape[2]], 
                                     strides=[1, bn_conv.shape[2]])(bn_conv)
    return tf.squeeze(pool, [2])

def tempo_block(inputs, filters, kernel_size, is_training, padding="same", activation=activations.relu):
    conv = layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  activation=activation,
                                  name=layer_name("conv2d"))(inputs)
    bn_conv = layers.BatchNormalization()(conv, 
                                                   training=is_training)
    print(bn_conv)
    pool = layers.MaxPooling2D(pool_size=[1, bn_conv.shape[2]], 
                                     strides=[1, bn_conv.shape[2]])(bn_conv)
    return tf.squeeze(pool, [2])

def frontend(x, is_training, y_input, n_filters):
    expand_input = tf.expand_dims(x, 3)
    normalized_input = layers.BatchNormalization()(expand_input,
                                                            training=is_training)
    input_pad_7 = tf.pad(normalized_input, [[0, 0], [3, 3], [0, 0], [0, 0]], 'CONSTANT')
    
    return [timbral_block(inputs=input_pad_7,
                          filters=int(n_filters*128),
                          kernel_size=[7, int(0.4 * y_input)],
                          is_training=is_training),
            timbral_block(inputs=input_pad_7,
                          filters=int(n_filters*128),
                          kernel_size=[7, int(0.7 * y_input)],
                          is_training=is_training),
            tempo_block(inputs=normalized_input,
                        filters=int(n_filters*32),
                        kernel_size=[128,1],
                        is_training=is_training),
            tempo_block(inputs=normalized_input,
                        filters=int(n_filters*32),
                        kernel_size=[64,1],
                        is_training=is_training),
            tempo_block(inputs=normalized_input,
                        filters=int(n_filters*32),
                        kernel_size=[32,1],
                        is_training=is_training)
    ]

def midend(front_end_output, is_training, n_filters):
    front_end_output = tf.expand_dims(front_end_output, 3)

    # conv layer 1
    front_end_pad = tf.pad(front_end_output, [[0, 0], [3, 3], [0, 0], [0, 0]], 'CONSTANT')
    conv1 = layers.Conv2D(filters=n_filters,
                                   kernel_size=[7, front_end_pad.shape[2]],
                                   padding="valid",
                                   activation=activations.relu,
                                   name=layer_name('conv2d'))(front_end_pad,
                                                              training=is_training)
    bn_conv1 = layers.BatchNormalization()(conv1, training=is_training)
    bn_conv1_t = tf.transpose(bn_conv1, [0, 1, 3, 2])

    # conv layer 2
    bn_conv1_pad = tf.pad(bn_conv1_t, [[0, 0], [3, 3], [0, 0], [0, 0]], 'CONSTANT')
    conv2 = layers.Conv2D(filters=n_filters,
                                   kernel_size=[7, bn_conv1_pad.shape[2]],
                                   padding="valid",
                                   activation=activations.relu,
                                   name=layer_name('conv2d'))(bn_conv1_pad,
                                                              training=is_training)
    bn_conv2 = layers.BatchNormalization()(conv2, training=is_training)
    conv2 = tf.transpose(bn_conv2, [0, 1, 3, 2])
    res_conv2 = tf.add(conv2, bn_conv1_t)

    # conv layer 3
    bn_conv2_pad = tf.pad(conv2, [[0, 0], [3, 3], [0, 0], [0, 0]], 'CONSTANT')
    conv3 = layers.Conv2D(filters=n_filters,
                                   kernel_size=[7, bn_conv2_pad.shape[2]],
                                   padding="valid",
                                   activation=activations.relu,
                                   name=layer_name('conv2d'))(bn_conv2_pad,
                                                              training=is_training)
    bn_conv3 = layers.BatchNormalization()(conv3, training=is_training)
    conv3 = tf.transpose(bn_conv3, [0, 1, 3, 2])
    res_conv3 = tf.add(conv3, res_conv2)

    return [front_end_output, bn_conv1_t, res_conv2, res_conv3]

def backend(feature_map, is_training, n_classes, output_units):
    # temporal pooling
    max_pool = tf.reduce_max(feature_map, axis=1)
    mean_pool, var_pool = tf.nn.moments(feature_map, axes=[1])
    tmp_pool = tf.concat([max_pool, mean_pool], 2)

    # penultimate dense layer
    flat_pool = layers.Flatten()(tmp_pool)
    flat_pool = layers.BatchNormalization()(flat_pool, training=is_training)
    flat_pool_dropout = layers.Dropout(rate=0.5)(flat_pool, training=is_training)
    dense = layers.Dense(units=output_units, 
                                  activation=activations.relu, 
                                  name=layer_name('dense'))(flat_pool_dropout)
    bn_dense = layers.BatchNormalization()(dense, training=is_training)
    dense_dropout = layers.Dropout(rate=0.5)(bn_dense, training=is_training)

    logits = layers.Dense(activation=None, units=n_classes, name=layer_name('dense'))(dense_dropout)

    return logits


def build_musicnn(x, is_training, n_classes, n_filters_frontend=1.6, n_filters_midend=64, n_units_backend=200):
    # frontend: musically-motivated CNN
    frontend_features_list = frontend(x, is_training, config.N_MELS, n_filters=n_filters_frontend)
    frontend_features = tf.concat(frontend_features_list, 2)

    # middle: dense layers
    midend_features_list = midend(frontend_features, is_training, n_filters_midend)
    midend_features = tf.concat(midend_features_list, 2)

    # backend: temporal pooling
    logits = backend(midend_features, is_training, n_classes, n_units_backend)
    return logits


def define_model(x, is_training, model, n_classes):
    if model == 'MTT_musicnn':
        return build_musicnn(x, is_training, n_classes, n_filters_midend=64, n_units_backend=200)
    elif model == 'MSD_musicnn':
        return build_musicnn(x, is_training, n_classes, n_filters_midend=64, n_units_backend=200)
    elif model == 'MSD_musicnn_big':
        return build_musicnn(x, is_training, n_classes, n_filters_midend=512, n_units_backend=500)       
    else:
        raise ValueError('Model not implemented!')


class ModelLayer(keras.Layer):
    def __init__(self, is_training, model, n_classes):
        super().__init__()
        self.is_training = is_training
        self.model = model
        self.n_classes = n_classes

    def call(self, x):
        return define_model(x, self.is_training, self.model, self.n_classes)


if __name__ == '__main__':
    model = 'MSD_musicnn_big'
# def load_musicnn_model(model: str='MSD_musicnn_big'):
    # from musicnn/musicnn/extractor.py
    if 'MTT' in model:
        labels = config.MTT_LABELS
    elif 'MSD' in model:
        labels = config.MSD_LABELS
    n_classes = len(labels)

    # n_frames is placeholder, we set some random value first
    n_frames = 187
    inputs = keras.Input(shape=(n_frames, config.N_MELS))
    model_outputs = ModelLayer(False, model, n_classes)(inputs)
    # model_outputs = define_model(inputs, False, model, n_classes)

    model_keras = keras.Model(inputs=inputs, outputs=model_outputs)
    keras.utils.plot_model(model_keras, 'musicnn_arch.png', show_shapes=True)

    ckpt_path = os.path.dirname(__file__) + '\\musicnn\\musicnn\\'+model+'\\'
    with open(ckpt_path + 'checkpoint', 'w') as ckpt_if:
        ckpt_if.write(f"model_checkpoint_path: {repr(ckpt_path)}\nall_model_checkpoint_paths: {repr(ckpt_path)}\n")

    checkpt_state = tf.train.list_variables(ckpt_path)
    print(type(checkpt_state))
    print(checkpt_state)
    # except:
    #     raise OSError('Unable to load model from musicnn repository (is it in the same directory as this script?)')
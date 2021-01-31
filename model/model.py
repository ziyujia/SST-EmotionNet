from __future__ import absolute_import, division, print_function


import keras
import keras.backend as K
from keras.layers import Input
from keras.layers.convolutional import Conv3D
from keras.layers.core import Activation, Dense, Dropout, Lambda
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import (AveragePooling3D, GlobalAveragePooling3D,
                                  MaxPooling3D)
from keras.models import Model
from keras.regularizers import l2


def DenseNet(input_shape=None, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1, nb_layers_per_block=-1,
             bottleneck=False, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, subsample_initial_block=False,
             include_top=True, weights=None, input_tensor=None,
             classes=10, activation='softmax'):

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `cifar10` '
                         '(pre-training on CIFAR-10).')

    if activation not in ['softmax', 'sigmoid']:
        raise ValueError('activation must be one of "softmax" or "sigmoid"')

    if activation == 'sigmoid' and classes != 1:
        raise ValueError(
            'sigmoid activation can only be used when classes = 1')

    img_input = Input(tensor=input_tensor, shape=input_shape)

    x = __create_dense_net(classes, img_input, include_top, depth, nb_dense_block,
                           growth_rate, nb_filter, nb_layers_per_block, bottleneck, reduction,
                           dropout_rate, weight_decay, subsample_initial_block, activation)

    model = Model(img_input, x, name='densenet')

    return model


def __conv_block(ip, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1e-4):

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)

    if bottleneck:
        inter_channel = nb_filter * 4

        x = Conv3D(inter_channel, (1, 1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
                   kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)

    x = Conv3D(nb_filter, (3, 3, 1), kernel_initializer='he_normal',
               padding='same', use_bias=False)(x)
    x = Conv3D(nb_filter, (1, 1, 3), kernel_initializer='he_normal',
               padding='same', use_bias=False)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def __dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1e-4,
                  grow_nb_filters=True, return_concat_list=False):

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x_list = [x]

    for i in range(nb_layers):
        cb = __conv_block(x, growth_rate, bottleneck,
                          dropout_rate, weight_decay)
        x_list.append(cb)

        x = concatenate([x, cb], axis=concat_axis)

        if grow_nb_filters:
            nb_filter += growth_rate

    if return_concat_list:
        return x, nb_filter, x_list
    else:
        return x, nb_filter


def __transition_block(ip, nb_filter, compression=1.0, weight_decay=1e-4):

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)
    x = Conv3D(int(nb_filter * compression), (1, 1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(x)

    return x


def __create_dense_net(nb_classes, img_input, include_top, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1,
                       nb_layers_per_block=-1, bottleneck=False, reduction=0.0, dropout_rate=None, weight_decay=1e-4,
                       subsample_initial_block=False, activation='softmax', attention=True, spatial_attention=True, temporal_attention=True):

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    if reduction != 0.0:
        assert reduction <= 1.0 and reduction > 0.0, 'reduction value must lie between 0.0 and 1.0'

    # layers in each dense block
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)  # Convert tuple to list

        assert len(nb_layers) == (nb_dense_block), 'If list, nb_layer is used as provided. ' \
                                                   'Note that list size must be (nb_dense_block)'
        final_nb_layer = nb_layers[-1]
        nb_layers = nb_layers[:-1]
    else:
        if nb_layers_per_block == -1:
            assert (
                depth - 4) % 3 == 0, 'Depth must be 3 N + 4 if nb_layers_per_block == -1'
            count = int((depth - 4) / 3)

            if bottleneck:
                count = count // 2

            nb_layers = [count for _ in range(nb_dense_block)]
            final_nb_layer = count
        else:
            final_nb_layer = nb_layers_per_block
            nb_layers = [nb_layers_per_block] * nb_dense_block

    # compute initial nb_filter if -1, else accept users initial nb_filter
    if nb_filter <= 0:
        nb_filter = 2 * growth_rate

    # compute compression factor
    compression = 1.0 - reduction

    # Initial convolution
    if subsample_initial_block:
        initial_kernel = (5, 5, 3)
        initial_strides = (2, 2, 1)
    else:
        initial_kernel = (3, 3, 1)
        initial_strides = (1, 1, 1)

    x = Conv3D(nb_filter, initial_kernel, kernel_initializer='he_normal', padding='same',
               strides=initial_strides, use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)
    x = Conv3D(nb_filter, initial_kernel, kernel_initializer='he_normal', padding='same',
               strides=initial_strides, use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)

    if subsample_initial_block:
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = __dense_block(x, nb_layers[block_idx], nb_filter, growth_rate, bottleneck=bottleneck,
                                     dropout_rate=dropout_rate, weight_decay=weight_decay)
        # add transition_block
        x = __transition_block(
            x, nb_filter, compression=compression, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)
        # add attention_block
        if attention:
            x = Attention_block(
                x, spatial_attention=spatial_attention, temporal_attention=temporal_attention)

    # The last dense_block does not have a transition_block
    x, nb_filter = __dense_block(x, final_nb_layer, nb_filter, growth_rate, bottleneck=bottleneck,
                                 dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling3D()(x)

    if include_top:
        x = Dense(nb_classes, activation=activation)(x)

    return x


def channel_wise_mean(x):
    mid = K.mean(x, axis=-1)
    return mid


def Attention_block(input_tensor, spatial_attention=True, temporal_attention=True):
    tem = input_tensor
    x = Lambda(channel_wise_mean)(input_tensor)
    x = keras.layers.Reshape([K.int_shape(input_tensor)[1], K.int_shape(
        input_tensor)[2], K.int_shape(input_tensor)[3], 1])(x)

    nbSpatial = K.int_shape(input_tensor)[1] * K.int_shape(input_tensor)[2]
    nbTemporal = K.int_shape(input_tensor)[-2]

    if spatial_attention:
        spatial = AveragePooling3D(
            pool_size=[1, 1, K.int_shape(input_tensor)[-2]])(x)
        spatial = keras.layers.Flatten()(spatial)
        spatial = Dense(nbSpatial)(spatial)
        spatial = Activation('sigmoid')(spatial)
        spatial = keras.layers.Reshape(
            [K.int_shape(input_tensor)[1], K.int_shape(input_tensor)[2], 1, 1])(spatial)

        tem = keras.layers.multiply([input_tensor, spatial])

    if temporal_attention:
        temporal = AveragePooling3D(pool_size=[K.int_shape(input_tensor)[
                                    1], K.int_shape(input_tensor)[2], 1])(x)
        temporal = keras.layers.Flatten()(temporal)
        temporal = Dense(nbTemporal)(temporal)
        temporal = Activation('sigmoid')(temporal)
        temporal = keras.layers.Reshape(
            [1, 1, K.int_shape(input_tensor)[-2], 1])(temporal)

        tem = keras.layers.multiply([temporal, tem])

    return tem


def sst_emotionnet(input_width, specInput_length, temInput_length, depth_spec, depth_tem, gr_spec, gr_tem, nb_dense_block, attention=True, spatial_attention=True, temporal_attention=True, nb_class=3):
    '''
    Model Input: [Spatial-Spectral Stream Input, Spatial-Temporal Stream Input]

    '''
    # Spatial-Spectral Stream
    specInput = Input([input_width, input_width, specInput_length, 1])
    x_s = __create_dense_net(img_input=specInput, depth=depth_spec, nb_dense_block=nb_dense_block,
                             growth_rate=gr_spec, nb_classes=nb_class, reduction=0.5, bottleneck=True, include_top=False, attention=attention, spatial_attention=spatial_attention, temporal_attention=temporal_attention)
    # Spatial-Temporal Stream
    temInput = Input([input_width, input_width, temInput_length, 1])
    x_t = __create_dense_net(img_input=temInput, depth=depth_tem, nb_dense_block=nb_dense_block,
                             growth_rate=gr_tem, nb_classes=nb_class, bottleneck=True, include_top=False, subsample_initial_block=True, attention=attention)

    y = keras.layers.concatenate([x_s, x_t], axis=-1)
    y = keras.layers.Dense(50)(y)
    y = keras.layers.Dropout(0.5)(y)
    if nb_class == 2:
        y = keras.layers.Dense(nb_class, activation='sigmoid')(y)
    else:
        y = keras.layers.Dense(nb_class, activation='softmax')(y)

    model = Model([specInput, temInput], y)
    return model

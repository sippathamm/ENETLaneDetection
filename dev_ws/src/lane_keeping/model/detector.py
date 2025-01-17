"""
Author: Sippawit Thammawiset
Date: 2024.08.08
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPool2D,
    Permute,
    ZeroPadding2D,
    BatchNormalization,
    PReLU,
    Add,
    SpatialDropout2D,
    UpSampling2D,
    ReLU,
    Rescaling,
    Conv2DTranspose,
    Activation,
    concatenate,
)
import numpy as np


def initial_block(inp):
    inp1 = inp
    conv = Conv2D(filters=13, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(inp)
    pool = MaxPool2D(pool_size=2)(inp1)
    concat = concatenate(inputs=[conv, pool])

    return concat


def encoder_bottleneck(inp, filters, name, downsample=False, dilated=False, dilation_rate=2, asymmetric=False,
                       drop_rate=0.1):
    reduce = filters // 4
    down = inp
    kernel_stride = 1

    # Downsample
    if downsample:
        kernel_stride = 2
        pad_activations = filters - list(inp.shape)[-1]
        down = MaxPool2D(pool_size=2)(down)
        down = Permute(dims=(1, 3, 2))(down)
        down = ZeroPadding2D(padding=((0, 0), (0, pad_activations)))(down)
        down = Permute(dims=(1, 3, 2))(down)

    # 1*1 Reduce
    x = Conv2D(filters=reduce, kernel_size=kernel_stride, strides=kernel_stride, padding='same', use_bias=False,
               kernel_initializer='he_normal', name=f'{name}_reduce')(inp)
    x = BatchNormalization(momentum=0.1)(x)
    x = PReLU(shared_axes=[1, 2])(x)

    # Conv
    if not dilated and not asymmetric:
        x = Conv2D(filters=reduce, kernel_size=3, padding='same', kernel_initializer='he_normal',
                   name=f'{name}_conv_reg')(x)
    elif dilated:
        x = Conv2D(filters=reduce, kernel_size=3, padding='same', dilation_rate=dilation_rate,
                   kernel_initializer='he_normal', name=f'{name}_reduce_dilated')(x)
    elif asymmetric:
        x = Conv2D(filters=reduce, kernel_size=(1, 5), padding='same', use_bias=False, kernel_initializer='he_normal',
                   name=f'{name}_asymmetric')(x)
        x = Conv2D(filters=reduce, kernel_size=(5, 1), padding='same', kernel_initializer='he_normal', name=name)(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = PReLU(shared_axes=[1, 2])(x)

    # 1*1 Expand
    x = Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False, kernel_initializer='he_normal',
               name=f'{name}_expand')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = SpatialDropout2D(rate=drop_rate)(x)

    concat = Add()([x, down])
    concat = PReLU(shared_axes=[1, 2])(concat)

    return concat


def decoder_bottleneck(inp, filters, name, upsample=False):
    reduce = filters // 4
    up = inp

    # Upsample
    if upsample:
        up = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same', use_bias=False,
                    kernel_initializer='he_normal', name=f'{name}_upsample')(up)
        up = UpSampling2D(size=2)(up)

    # 1*1 Reduce
    x = Conv2D(filters=reduce, kernel_size=1, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal',
               name=f'{name}_reduce')(inp)
    x = BatchNormalization(momentum=0.1)(x)
    x = PReLU(shared_axes=[1, 2])(x)

    # Conv
    if not upsample:
        x = Conv2D(filters=reduce, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                   name=f'{name}_conv_reg')(x)
    else:
        x = Conv2DTranspose(filters=reduce, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal',
                            name=f'{name}_transpose')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = PReLU(shared_axes=[1, 2])(x)

    # 1*1 Expand
    x = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same', use_bias=False,
               kernel_initializer='he_normal', name=f'{name}_expand')(x)
    x = BatchNormalization(momentum=0.1)(x)

    concat = Add()([x, up])
    concat = ReLU()(concat)

    return concat


class Detector:
    def __init__(self, input_shape: tuple[int, int, int], name: str = 'Detector'):
        self.name: str = name
        self.model: Model = self.build_model(input_shape=input_shape, name=self.name)

    def __call__(self, x: np.ndarray, verbose: int = 1) -> np.ndarray:
        return self.model.predict(x, verbose=verbose)[0]

    @staticmethod
    def build_model(input_shape: tuple[int, int, int], name: str = 'Detector') -> Model:
        input_width = input_shape[0]
        input_height = input_shape[1]
        input_channels = input_shape[2]

        inputs = Input(shape=(input_height, input_width, input_channels))
        rescale = Rescaling(scale=1. / 127.5, offset=-1)(inputs)
        enc = initial_block(rescale)

        # Bottleneck 1.0
        enc = encoder_bottleneck(enc, 64, name='enc1.0', downsample=True, drop_rate=0.01)
        enc = encoder_bottleneck(enc, 64, name='enc1.1', drop_rate=0.01)
        enc = encoder_bottleneck(enc, 64, name='enc1.2', drop_rate=0.01)
        enc = encoder_bottleneck(enc, 64, name='enc1.3', drop_rate=0.01)
        enc = encoder_bottleneck(enc, 64, name='enc1.4', drop_rate=0.01)

        # Bottleneck 2.0
        enc = encoder_bottleneck(enc, 128, name='enc2.0', downsample=True)
        enc = encoder_bottleneck(enc, 128, name='enc2.1')
        enc = encoder_bottleneck(enc, 128, name='enc2.2', dilated=True, dilation_rate=2)
        enc = encoder_bottleneck(enc, 128, name='enc2.3', asymmetric=True)
        enc = encoder_bottleneck(enc, 128, name='enc2.4', dilated=True, dilation_rate=4)
        enc = encoder_bottleneck(enc, 128, name='enc2.5')
        enc = encoder_bottleneck(enc, 128, name='enc2.6', dilated=True, dilation_rate=8)
        enc = encoder_bottleneck(enc, 128, name='enc2.7', asymmetric=True)
        enc = encoder_bottleneck(enc, 128, name='enc2.8', dilated=True, dilation_rate=16)

        # Bottleneck 3.0
        enc = encoder_bottleneck(enc, 128, name='enc3.0')
        enc = encoder_bottleneck(enc, 128, name='enc3.1', dilated=True, dilation_rate=2)
        enc = encoder_bottleneck(enc, 128, name='enc3.2', asymmetric=True)
        enc = encoder_bottleneck(enc, 128, name='enc3.3', dilated=True, dilation_rate=4)
        enc = encoder_bottleneck(enc, 128, name='enc3.4')
        enc = encoder_bottleneck(enc, 128, name='enc3.5', dilated=True, dilation_rate=8)
        enc = encoder_bottleneck(enc, 128, name='enc3.6', asymmetric=True)
        enc = encoder_bottleneck(enc, 128, name='enc3.7', dilated=True, dilation_rate=16)

        # Bottleneck 4.0
        dec = decoder_bottleneck(enc, 64, name='dec4.0', upsample=True)
        dec = decoder_bottleneck(dec, 64, name='dec4.1')
        dec = decoder_bottleneck(dec, 64, name='dec4.2')

        # Bottleneck 5.0
        dec = decoder_bottleneck(dec, 16, name='dec5.0', upsample=True)
        dec = decoder_bottleneck(dec, 16, name='dec5.1')

        dec = Conv2DTranspose(filters=1, kernel_size=2, strides=2, padding='same',
                              kernel_initializer='he_normal')(dec)
        outputs = Activation(activation='sigmoid')(dec)

        return Model(inputs=[inputs], outputs=[outputs], name=name)

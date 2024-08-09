"""
Author: Sippawit Thammawiset
Date: 2024.08.08
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Rescaling,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    MaxPool2D,
    concatenate,
    Dropout,
    SpatialDropout2D,
    Flatten,
    Dense,
    Activation
)
import numpy as np


class Predictor:
    def __init__(self, input_shape: tuple[int, int, int], name: str = 'Predictor'):
        self.name: str = name
        self.model: Model = self.build_model(input_shape=input_shape, name=self.name)

    def __call__(self, x: np.ndarray, verbose: int = 1) -> np.ndarray:
        return self.model.predict(x, verbose=verbose)[0]

    @staticmethod
    def build_model(input_shape: tuple[int, int, int], name: str = 'Predictor') -> Model:
        input_width: int = input_shape[0]
        input_height: int = input_shape[1]
        input_channels: int = input_shape[2]

        inputs = Input(shape=(input_height, input_width, input_channels))
        rescale = Rescaling(scale=1. / 127.5, offset=-1)(inputs)

        a = Conv2D(filters=16, kernel_size=5, strides=2, activation='elu',
                   kernel_initializer='he_normal')(rescale)
        a = Dropout(rate=0.1)(a)
        a = Conv2D(filters=16, kernel_size=5, strides=2, activation='elu',
                   kernel_initializer='he_normal')(a)
        a = Dropout(rate=0.1)(a)
        a = Conv2D(filters=32, kernel_size=5, strides=2, activation='elu',
                   kernel_initializer='he_normal')(a)
        a = Dropout(rate=0.1)(a)
        a = Conv2D(filters=64, kernel_size=3, strides=1, activation='elu',
                   kernel_initializer='he_normal')(a)
        a = Dropout(rate=0.1)(a)
        a = Conv2D(filters=64, kernel_size=3, strides=1)(a)
        a = MaxPooling2D(pool_size=2, strides=2, padding='same')(a)

        b = Flatten()(a)
        b = Dropout(rate=0.2)(b)
        b = BatchNormalization()(b)
        b = Activation('elu')(b)

        c = Dense(units=512)(b)
        c = Dense(units=128)(c)
        c = Dense(units=32)(c)
        outputs = Dense(units=1)(c)

        return Model(inputs=[inputs], outputs=[outputs], name=name)

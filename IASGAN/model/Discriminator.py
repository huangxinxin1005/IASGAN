from keras.optimizers import rmsprop_v2
from keras.initializers.initializers_v2 import RandomNormal, Zeros
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Concatenate, Reshape
from keras import regularizers
from keras.layers import BatchNormalization

def global_conv_block(x, out_dim, kernel_size):
    conv_l1 = Conv2D(out_dim, kernel_size=(kernel_size[0], 1),
                            padding='valid',
                            kernel_initializer=RandomNormal(stddev=0.01),
                            bias_initializer=Zeros(),
                            kernel_regularizer=regularizers.l2(0.01))(x)
    conv_l2 = Conv2D(out_dim, kernel_size=(1, kernel_size[1]),
                            padding='valid',
                            kernel_initializer=RandomNormal(stddev=0.01),
                            bias_initializer=Zeros(),
                            kernel_regularizer=regularizers.l2(0.01))(conv_l1)
    conv_r1 = Conv2D(out_dim, kernel_size=(1, kernel_size[1]),
                            padding='valid',
                            kernel_initializer=RandomNormal(stddev=0.01),
                            bias_initializer=Zeros(),
                            kernel_regularizer=regularizers.l2(0.01))(x)
    conv_r2 = Conv2D(out_dim, kernel_size=(kernel_size[0], 1),
                            padding='valid',
                            kernel_initializer=RandomNormal(stddev=0.01),
                            bias_initializer=Zeros(),
                            kernel_regularizer=regularizers.l2(0.01))(conv_r1)

    add_layer = conv_l2 + conv_r2
    return add_layer

def define_discriminator(image_shape):
    init = RandomNormal(stddev=0.02)
    in_src_image = Input(shape=image_shape)

    d1 = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(in_src_image)
    d1 = LeakyReLU(alpha=0.2)(d1)

    d2 = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d1)
    d2 = BatchNormalization()(d2)
    d2 = LeakyReLU(alpha=0.2)(d2)
    #d2 = Dropout(0.1)(d2)

    d3 = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d2)
    d3 = BatchNormalization()(d3)
    d3 = LeakyReLU(alpha=0.2)(d3)
    #d3 = Dropout(0.1)(d3)

    d4 = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d3)
    d4 = BatchNormalization()(d4)
    d4 = LeakyReLU(alpha=0.2)(d4)
    #d4 = Dropout(0.1)(d4)

    d5 = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d4)
    d5 = BatchNormalization()(d5)
    d5 = LeakyReLU(alpha=0.2)(d5)
    #d5 = Dropout(0.1)(d5)

    d6 = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d5)
    d6 = BatchNormalization()(d6)
    d6 = LeakyReLU(alpha=0.2)(d6)
    #d6 = Dropout(0.1)(d6)

    input = Reshape((1, -1))(in_src_image)
    out1 = Reshape((1, -1))(d1)
    out2 = Reshape((1, -1))(d2)
    out3 = Reshape((1, -1))(d3)
    out4 = Reshape((1, -1))(d4)
    out5 = Reshape((1, -1))(d5)
    out6 = Reshape((1, -1))(d6)
    output = Concatenate()([input, out1, out2, out3, out4, out5, out6])
    model = Model(in_src_image, output)
    model.summary()
    opt = rmsprop_v2.RMSprop(learning_rate=0.0001)
    model.compile(loss='mae', optimizer=opt)
    return model
# imports
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.losses import binary_crossentropy


# function making the encoder blocks for UNet
# takes inputs, number of filters, dropout probability, and a boolean variable for max pooling as input
def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
    # Add 2 Conv Layers with relu activation and HeNormal initialization
    # "same" padding pads the input to conv layer such that the output has the same height and width to avoid size reduction
    conv = Conv2D(n_filters,
                  3,  # kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(inputs)
    conv = Conv2D(n_filters,
                  3,  # kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(conv)

    # batch normalization to normalize the output of the last layer
    conv = BatchNormalization()(conv, training=False)

    # dropout to prevent over fitting
    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    # condition max pooling layer
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
    else:
        next_layer = conv

    # skip connection will be input to the decoder layer
    skip_connection = conv

    return next_layer, skip_connection


# function makes decoder block
# takes previous layer, skip connection, and number of filters as input
def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32):
    # transpose convolution layer to increase the size of the image
    up = Conv2DTranspose(
        n_filters,
        (3, 3),  # Kernel size
        strides=(2, 2),
        padding='same')(prev_layer_input)

    # concatenate skip connection from previous block
    merge = concatenate([up, skip_layer_input], axis=3)

    # 2 conv Layers with relu activation and HeNormal initialization
    conv = Conv2D(n_filters,
                  3,  # kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(merge)
    conv = Conv2D(n_filters,
                  3,  # kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(conv)
    return conv


# function making 4-layer U-Net
# takes input size, number of filters, and number of classes as input
def UNet4(input_size=(128, 128, 3), n_filters=32, n_classes=3):
    # size of image
    inputs = Input(input_size)
    # encoder blocks
    cblock1 = EncoderMiniBlock(inputs, n_filters, dropout_prob=0, max_pooling=True)
    cblock2 = EncoderMiniBlock(cblock1[0], n_filters * 2, dropout_prob=0, max_pooling=True)
    cblock3 = EncoderMiniBlock(cblock2[0], n_filters * 4, dropout_prob=0, max_pooling=True)
    cblock4 = EncoderMiniBlock(cblock3[0], n_filters * 8, dropout_prob=0.3, max_pooling=True)
    cblock5 = EncoderMiniBlock(cblock4[0], n_filters * 16, dropout_prob=0.3, max_pooling=False)

    # decoder blocks
    ublock6 = DecoderMiniBlock(cblock5[0], cblock4[1], n_filters * 8)
    ublock7 = DecoderMiniBlock(ublock6, cblock3[1], n_filters * 4)
    ublock8 = DecoderMiniBlock(ublock7, cblock2[1],  n_filters * 2)
    ublock9 = DecoderMiniBlock(ublock8, cblock1[1], n_filters)

    # final convolutional layer for output
    conv9 = Conv2D(n_filters,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(ublock9)

    conv10 = Conv2D(n_classes, 1, padding='same')(conv9)

    # define the model
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model


# function making 3-layer U-Net
# takes input size, number of filters, and number of classes as input
def UNet3(input_size=(128, 128, 3), n_filters=32, n_classes=3):
    # size of image
    inputs = Input(input_size)
    # encoder blocks
    cblock1 = EncoderMiniBlock(inputs, n_filters, dropout_prob=0, max_pooling=True)
    cblock2 = EncoderMiniBlock(cblock1[0], n_filters * 2, dropout_prob=0, max_pooling=True)
    cblock3 = EncoderMiniBlock(cblock2[0], n_filters * 4, dropout_prob=0, max_pooling=True)
    cblock4 = EncoderMiniBlock(cblock3[0], n_filters * 8, dropout_prob=0.3, max_pooling=False)
    #cblock5 = EncoderMiniBlock(cblock4[0], n_filters * 16, dropout_prob=0.3, max_pooling=False)

    # decoder blocks
    #ublock6 = DecoderMiniBlock(cblock5[0], cblock4[1], n_filters * 8)
    ublock7 = DecoderMiniBlock(cblock4[0], cblock3[1], n_filters * 4)
    ublock8 = DecoderMiniBlock(ublock7, cblock2[1],  n_filters * 2)
    ublock9 = DecoderMiniBlock(ublock8, cblock1[1], n_filters)

    # final convolutional layer for output
    conv9 = Conv2D(n_filters,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(ublock9)

    conv10 = Conv2D(n_classes, 1, padding='same')(conv9)

    # define the model
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model


# function making 2-layer U-Net
# takes input size, number of filters, and number of classes as input
def UNet2(input_size=(128, 128, 3), n_filters=32, n_classes=3):
    # size of image
    inputs = Input(input_size)

    # encoder blocks
    cblock1 = EncoderMiniBlock(inputs, n_filters, dropout_prob=0, max_pooling=True)
    cblock2 = EncoderMiniBlock(cblock1[0], n_filters * 2, dropout_prob=0, max_pooling=True)
    cblock3 = EncoderMiniBlock(cblock2[0], n_filters * 4, dropout_prob=0, max_pooling=False)
    #cblock4 = EncoderMiniBlock(cblock3[0], n_filters * 8, dropout_prob=0.3, max_pooling=True)
    #cblock5 = EncoderMiniBlock(cblock4[0], n_filters * 16, dropout_prob=0.3, max_pooling=False)

    # decoder blocks
    #ublock6 = DecoderMiniBlock(cblock5[0], cblock4[1], n_filters * 8)
    #ublock7 = DecoderMiniBlock(ublock6, cblock3[1], n_filters * 4)
    ublock8 = DecoderMiniBlock(cblock3[0], cblock2[1],  n_filters * 2)
    ublock9 = DecoderMiniBlock(ublock8, cblock1[1], n_filters)

    # final convolutional layer for output
    conv9 = Conv2D(n_filters,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(ublock9)

    conv10 = Conv2D(n_classes, 1, padding='same')(conv9)

    # define the model
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model

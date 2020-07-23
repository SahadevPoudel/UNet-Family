import keras
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.layers import Input, merge, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras.layers.noise import GaussianDropout

import numpy as np
smooth = 1.
dropout_rate = 0.5
act = "relu"

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# Custom loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

# Evaluation metric: IoU
def compute_iou(im1, im2):
    overlap = (im1>0.5) * (im2>0.5)
    union = (im1>0.5) + (im2>0.5)
    return overlap.sum()/float(union.sum())

# Evaluation metric: Dice
def compute_dice(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1>0.5).astype(np.bool)
    im2 = np.asarray(im2>0.5).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):

    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    x = Dropout(dropout_rate, name='dp'+stage+'_1')(x)
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(dropout_rate, name='dp'+stage+'_2')(x)

    return x


def UNetPlusPlus(img_rows, img_cols, color_type=3, num_class=1, deep_supervision=True):
    nb_filter = [32, 64, 128, 256, 512]
    import efficientnet.keras as efn
    from keras.models import Model
    from keras.layers.convolutional import Conv2D
    from keras.layers import LeakyReLU, Add, Input, MaxPool2D, UpSampling2D, concatenate, Conv2DTranspose, \
        BatchNormalization, Dropout
    base_model = efn.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    input_model = base_model.input
    # Handle Dimension Ordering for different backends
    global bn_axis
    bn_axis = 3
    conv6_1 = base_model.get_layer('top_activation').output
    conv5_1 = base_model.get_layer('block6a_expand_activation').output
    conv4_1 = base_model.get_layer('block4a_expand_activation').output
    conv3_1 = base_model.get_layer('block3a_expand_activation').output
    conv2_1 = base_model.get_layer('block2a_expand_activation').output

    print(conv6_1.shape)  # 7*7
    print(conv5_1.shape)  # 14*14
    print(conv4_1.shape)  # 28*28
    print(conv3_1.shape)  # 56*56
    print(conv2_1.shape)  # 112*112

    conv1_1 = standard_unit(input_model, stage='11', nb_filter=nb_filter[0])
    # pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    # conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    # pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])

    #     conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    #     pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])

    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])

    #     conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    #     pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])

    #     conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])
    conv5_1_upsampling_16 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(16, 16), name='conv5_1_upsampling_16',
                                            padding='same')(conv5_1)  # changes made

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])
    conv4_2_upsampling_8 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(8, 8), name='conv4_2_upsampling_8',
                                           padding='same')(conv4_2)  # changes made

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])
    conv3_3_upsampling_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(4, 4), name='conv3_3_upsampling_4',
                                           padding='same')(conv3_3)  # changes made

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])
    conv2_4_upsampling_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='conv2_4_upsampling_2',
                                           padding='same')(conv2_4)  # changes made

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_1', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
    nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_2', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
    nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_4', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_5)
    # changes code
    nested_conv2_4_upsampling_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_5',
                                         kernel_initializer='he_normal',
                                         padding='same', kernel_regularizer=l2(1e-4))(conv2_4_upsampling_2)
    nested_conv3_3_upsampling_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_6',
                                         kernel_initializer='he_normal',
                                         padding='same', kernel_regularizer=l2(1e-4))(conv3_3_upsampling_4)
    nested_conv4_2_upsampling_8 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_7',
                                         kernel_initializer='he_normal',
                                         padding='same', kernel_regularizer=l2(1e-4))(conv4_2_upsampling_8)
    nested_conv5_1_upsampling_16 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_8',
                                          kernel_initializer='he_normal',
                                          padding='same', kernel_regularizer=l2(1e-4))(conv5_1_upsampling_16)
    nestnet_output_all = keras.layers.Average()(
        [nestnet_output_1, nestnet_output_2, nestnet_output_3, nestnet_output_4, nested_conv2_4_upsampling_2,
         nested_conv3_3_upsampling_4, nested_conv4_2_upsampling_8, nested_conv5_1_upsampling_16])

    model = Model(inputs=[input_model], outputs=[nestnet_output_all])
    #     if deep_supervision:
    #         model = Model(input=img_input, output=[nestnet_output_1,
    #                                                nestnet_output_2,
    #                                                nestnet_output_3,
    #                                                nestnet_output_4,
    #                                                nested_conv2_4_upsampling_2,
    #                                                nested_conv3_3_upsampling_4,
    #                                                nested_conv4_2_upsampling_8,
    #                                                nested_conv5_1_upsampling_16
    #                                                ])
    #     else:
    #         model = Model(input=img_input, output=[nestnet_output_4])

    return model

if __name__ == '__main__':
    model = UNetPlusPlus(224, 224, 3)
    model.summary()

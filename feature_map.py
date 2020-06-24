
from keras.models import load_model
from tensorflow import nn
from keras.backend import shape
from keras.layers import Dropout
from keras import backend as K
smooth = 1
import numpy as np
import tensorflow as tf
def dice_coef(y_true, y_pred):# y_true为真实准确值，y_pred为预测值
    y_true_f = K.flatten(y_true)# 捋直
    y_pred_f = K.flatten(y_pred)# 捋直
    # K.sum不加axi（指定方向求和，返回对应方向向量）,则为全元素求和，返回一个数字
    intersection = K.sum(y_true_f * y_pred_f)# 求预测准确的结果（真实准确值和预测值的交集）
    # 原始公式：（2*预测准确值）/（真实准确值+预测值），越大效果越好
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
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
class FixedDropout(Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        return tuple([shape(inputs)[i] if sh is None else sh for i, sh in enumerate(self.noise_shape)])

customObjects = {
    'swish': nn.swish,
    'FixedDropout': FixedDropout,
    'dice_coef':dice_coef,
    'mean_iou':mean_iou
}
model = load_model('check.h5', custom_objects=customObjects)
#print(model.summary())
from keras.preprocessing import image
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from numpy import expand_dims
from matplotlib import pyplot
from keras.models import Model
model = Model(inputs=model.inputs, outputs=model.layers[5].output)
#model.summary()
# load the image with the required shape
img = load_img('/home/poudelas/Documents/unet-master/data1/Kvasir-SEG/train/image/cju0qkwl35piu0993l0dewei2.jpg', target_size=(224, 224))
# convert the image to an array
img = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
#img = preprocess_input(img)
# get feature map for first hidden layer
feature_maps = model.predict(img)
# plot all 64 maps in an 8x8 squares
ix = 1
result=[]
for _ in range(32):
         result.append(feature_maps[0, :, :, ix - 1])
         ix += 1
# show the figure
image = np.mean(result,axis=0)
pyplot.imshow(image,cmap='gray')
pyplot.show()
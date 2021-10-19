from tensorflow.keras.applications import VGG16, MobileNetV2
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess
from tensorflow.keras.models import Model

def get_vgg():

    target_shape = (224,224,3)
    vgg = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=target_shape,
        pooling='avg')

    vgg.trainable = False
    
    
    input_layer = Input(target_shape, dtype = tf.uint8)
    y = tf.cast(input_layer, tf.float32)
    y = vgg16_preprocess(y)

    y = vgg(y)
    model = Model(inputs=input_layer, outputs=y)
    
    return model


def get_mobilenet():

    target_shape = (224,224,3)
    mobilenet = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=target_shape,
        pooling='avg')

    mobilenet.trainable = False
    
    
    input_layer = Input(target_shape, dtype = tf.uint8)
    y = tf.cast(input_layer, tf.float32)
    y = mobilenet_v2_preprocess(y)

    y = mobilenet(y)
    model = Model(inputs=input_layer, outputs=y)
    
    return model

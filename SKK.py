from tensorflow.keras.layers import *
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.metrics import *
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Concatenate
import tensorflow as tf 
def SKK(inputs, channel, code, ratio=8):

    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    
    conv1 = Conv2D(channel, (5, 5), padding='same', name='SK_conv_'+code+"_1")(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    
    conv2 = Conv2D(channel, (3, 3), padding='same', name='SK_conv_'+code+"_2")(inputs)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    
    conv_unite = Add()([conv1,conv2])

       
    avg_pool = GlobalAveragePooling2D()(conv_unite)
    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    
    max_pool = GlobalMaxPooling2D()(conv_unite)
    
    max_pool = Reshape((1,1,channel))(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    
    embedding = Add()([max_pool,avg_pool])
    embedding = Gamma()(embedding)
    embedding = Activation('sigmoid')(embedding)
    
    res = Lambda(lambda x:1-x)(embedding)
    
    if K.image_data_format() == "channels_first":
        embedding = Permute((3, 1, 2))(embedding)
        res = Permute((3, 1, 2))(res)
    
    conv1 = multiply([conv1, embedding])
    
    conv2 = multiply([conv2, res])
    
    ans = Add()([conv1,conv2])
    
    return ans
    
 
 
class Focal(tf.keras.layers.Layer):

    def __init__(self):
        super(Gamma, self).__init__()

    def build(self, input_shape):
        initializer = tf.keras.initializers.Ones()

        self.w = self.add_weight(
            shape=(1,1),
            initializer=initializer,
            trainable=True,
            constraint= tf.keras.constraints.NonNeg())

    def call(self, inputs):
        inputs = tf.clip_by_value(inputs, tf.keras.backend.epsilon(), 1.)
        return (inputs)**self.w

import tensorflow as tf
import numpy as np

from network.resblock import resBlock

class ResFCN256(tf.keras.Model):
    def __init__(self, resolution_inp=256, resolution_op=256, channel=3, name='resfcn256'):
        super(ResFCN256, self).__init__(name=name)
        self.channel = channel
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        
        # Define regularizer
        self.regularizer = tf.keras.regularizers.l2(0.0002)
        
        # Define layers
        size = 16
        self.conv1 = tf.keras.layers.Conv2D(size, kernel_size=4, strides=1, padding='SAME',
                                          kernel_regularizer=self.regularizer)
        
        # Encoder layers
        self.encoder_layers = []
        sizes = [size * mult for mult in [2, 2, 4, 4, 8, 8, 16, 16, 32, 32]]
        strides = [2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
        for s, stride in zip(sizes, strides):
            self.encoder_layers.append(
                lambda x, s=s, stride=stride: resBlock(x, s, kernel_size=4, stride=stride))
        
        # Decoder layers
        self.decoder_layers = []
        sizes = [size * mult for mult in [32, 16, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 1, 1]]
        strides = [1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1]
        for s, stride in zip(sizes, strides):
            self.decoder_layers.append(
                tf.keras.layers.Conv2DTranspose(s, kernel_size=4, strides=stride, padding='SAME',
                                              kernel_regularizer=self.regularizer))
        
        # Final layers
        self.conv_final1 = tf.keras.layers.Conv2DTranspose(3, kernel_size=4, strides=1, padding='SAME',
                                                          kernel_regularizer=self.regularizer)
        self.conv_final2 = tf.keras.layers.Conv2DTranspose(3, kernel_size=4, strides=1, padding='SAME',
                                                          kernel_regularizer=self.regularizer)
        self.conv_final3 = tf.keras.layers.Conv2DTranspose(3, kernel_size=4, strides=1, padding='SAME',
                                                          activation='sigmoid',
                                                          kernel_regularizer=self.regularizer)
        
    def call(self, x, training=True):
        # Initial conv
        x = self.conv1(x)
        
        # Encoder
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Decoder
        for layer in self.decoder_layers:
            x = layer(x)
            if training:
                x = tf.keras.layers.BatchNormalization()(x)
            x = tf.nn.relu(x)
        
        # Final convolutions
        x = self.conv_final1(x)
        x = self.conv_final2(x)
        x = self.conv_final3(x)
        
        return x
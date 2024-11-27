import tensorflow as tf
import numpy as np

def resBlock(x, num_outputs, 
             kernel_size=4, 
             stride=1, 
             activation_fn=tf.nn.relu, 
             use_batch_norm=True):
    assert num_outputs % 2 == 0  # num_outputs must be divided by channel_factor(2 here)
    
    shortcut = x
    if stride != 1 or x.shape[-1] != num_outputs:
        shortcut = tf.keras.layers.Conv2D(
            num_outputs, kernel_size=1, strides=stride,
            padding='SAME', use_bias=False)(shortcut)
    
    x = tf.keras.layers.Conv2D(
        num_outputs//2, kernel_size=1, strides=1,
        padding='SAME', use_bias=False)(x)
    x = tf.keras.layers.Conv2D(
        num_outputs//2, kernel_size=kernel_size, strides=stride,
        padding='SAME', use_bias=False)(x)
    x = tf.keras.layers.Conv2D(
        num_outputs, kernel_size=1, strides=1,
        padding='SAME', use_bias=False)(x)
    x = tf.keras.layers.Add()([shortcut, x])
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = activation_fn(x)
    
    return x
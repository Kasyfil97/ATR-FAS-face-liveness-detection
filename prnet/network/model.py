import os
import tensorflow as tf
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class PRNet:
    def __init__(self, resolution_inp=256, resolution_op=256):
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.max_pos = resolution_inp * 1.1
        
        # Model akan di-load saat restore dipanggil
        self.model = None
        self.serving_fn = None
    
    def restore(self, model_path):
        self.model = tf.saved_model.load(model_path)
        self.serving_fn = self.model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    def predict(self, image):
        if self.serving_fn is None:
            raise ValueError("Model belum di-load. Panggil restore() terlebih dahulu.")
        
        # Prepare input
        if len(image.shape) == 3:
            image = image[np.newaxis, ...]
        
        # Convert ke tensor
        input_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        
        # Predict
        output = self.serving_fn(input_image=input_tensor)
        prediction = output['prediction'].numpy()
        prediction = np.squeeze(prediction)
        
        return prediction * self.max_pos
    
    def predict_batch(self, images):
        pos = self.network(images, training=False)
        return pos.numpy() * self.MaxPos
import os
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from django.conf import settings

# Define custom objects (ArcFace layer and arcface_loss function)
def normalize_l2(x, axis=1):
    return x / tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis, keepdims=True) + 1e-10)

class ArcFace(tf.keras.layers.Layer):
    def __init__(self, num_classes, s=30.0, m=0.3, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.regularizer = regularizer

    def build(self, input_shape):
        self.W = self.add_weight(name="arcface_W",
                                shape=(input_shape[-1], self.num_classes),
                                initializer="glorot_uniform",
                                regularizer=self.regularizer,
                                trainable=True)
        super(ArcFace, self).build(input_shape)

    def call(self, embeddings, labels):
        embeddings_norm = normalize_l2(embeddings, axis=1)
        weights_norm = normalize_l2(self.W, axis=0)
        cosine = tf.matmul(embeddings_norm, weights_norm)
        return cosine * self.s
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "s": self.s,
            "m": self.m,
            "regularizer": tf.keras.regularizers.serialize(self.regularizer)
        })
        return config

def arcface_loss(y_true, y_pred):
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

# Load the model
MODEL_DIR = os.path.join(settings.BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "cat_dog_model.keras")

try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"ArcFace": ArcFace, "arcface_loss": arcface_loss}
    )
    # Get the embedding layer (assuming it's the third-to-last layer)
    embedding_model = tf.keras.Model(
        inputs=model.inputs[0],  # Image input
        outputs=model.layers[-3].output  # Embedding layer output
    )
    print("ArcFace model loaded successfully")
except Exception as e:
    model = None
    embedding_model = None
    print(f"Error loading ArcFace model: {e}")

def preprocess_image(image):
    """Preprocess image for the ArcFace model"""
    try:
        # Resize to the expected input size
        resized = cv2.resize(image, (224, 224))
        # Convert to RGB if it's in BGR format
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Convert to float and normalize
        img_array = img_to_array(resized)
        img_array = img_array / 255.0
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def extract_features(image):
    """Extract features from an image using the ArcFace model"""
    if embedding_model is None:
        print("ArcFace model is not loaded.")
        return None

    try:
        # Preprocess the image
        preprocessed = preprocess_image(image)
        if preprocessed is None:
            return None
        
        # Extract embeddings
        # For inference, we need to provide a dummy label, but it's not used
        dummy_label = np.zeros((1,), dtype=np.int32)
        
        # Get embeddings from the model
        embeddings = embedding_model.predict(preprocessed, verbose=0)
        
        # Normalize embeddings
        normalized = embeddings / (np.linalg.norm(embeddings) + 1e-7)
        
        return normalized.flatten().tolist()
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def compare_features(features1, features2):
    """Compare two feature vectors and return similarity score"""
    try:
        if not features1 or not features2:
            return 0.0

        f1, f2 = np.array(features1), np.array(features2)
        if f1.size == 0 or f2.size == 0:
            return 0.0

        # Compute cosine similarity
        similarity = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-7)
        return float(similarity)
    except Exception as e:
        print(f"Error comparing features: {e}")
        return 0.0

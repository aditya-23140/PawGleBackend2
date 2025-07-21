import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# --- Part 1: Your Custom ArcFace Code ---
# We need to include the custom layer definition directly in the app
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

# --- Part 2: Model Loading and Helper Functions ---
MODEL_PATH = "cat_dog_model.keras"

# Load the full model with custom objects
try:
    full_model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"ArcFace": ArcFace, "arcface_loss": arcface_loss}
    )
    # Create a model to extract embeddings (output of the third-to-last layer)
    embedding_model = tf.keras.Model(
        inputs=full_model.inputs[0],
        outputs=full_model.layers[-3].output
    )
    print("✅ ArcFace model and embedding model loaded successfully.")
except Exception as e:
    embedding_model = None
    print(f"❌ Error loading ArcFace model: {e}")

def preprocess_image(image_pil):
    """Preprocesses a PIL image for the model."""
    image_np = np.array(image_pil).astype('uint8')
    resized = cv2.resize(image_np, (224, 224))
    if len(resized.shape) == 3 and resized.shape[2] == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR) # OpenCV uses BGR
    img_array = img_to_array(resized)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def extract_features(image_pil):
    """Extracts a normalized feature vector from a PIL image."""
    if embedding_model is None: return None
    preprocessed = preprocess_image(image_pil)
    embeddings = embedding_model.predict(preprocessed, verbose=0)
    normalized = embeddings / (np.linalg.norm(embeddings) + 1e-7)
    return normalized.flatten()

# --- Part 3: The Main Gradio Function ---
def compare_images(image1_pil, image2_pil):
    """The main function for the Gradio interface."""
    if image1_pil is None or image2_pil is None:
        return "Please upload two images."
        
    features1 = extract_features(image1_pil)
    features2 = extract_features(image2_pil)
    
    if features1 is None or features2 is None:
        return "Error processing one of the images."

    # Compute cosine similarity
    similarity = np.dot(features1, features2)
    
    return f"Similarity Score: {similarity:.4f}\n(A score closer to 1.0 means the pets are more similar)"

# --- Part 4: Launch the Gradio Interface ---
iface = gr.Interface(
    fn=compare_images,
    inputs=[
        gr.Image(type="pil", label="Image 1"),
        gr.Image(type="pil", label="Image 2")
    ],
    outputs=gr.Textbox(label="Result"),
    title="🐾 Pet Face Similarity Checker",
    description="Upload two images of pets to see how similar they are based on facial features. Powered by ArcFace.",
    allow_flagging="never"
)

iface.launch()
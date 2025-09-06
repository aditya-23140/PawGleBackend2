import os
import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

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
            "regularizer": tf.keras.regularizers.serialize(self.regularizer) if self.regularizer else None
        })
        return config

def arcface_loss(y_true, y_pred):
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

# Global variables for models
model = None
embedding_model = None

def load_models():
    """Load the ArcFace model and embedding model"""
    global model, embedding_model
    
    MODEL_PATH = "cat_dog_model.keras"  # Place your model file in the repository root
    
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
        return True
    except Exception as e:
        print(f"Error loading ArcFace model: {e}")
        return False

def preprocess_image(image):
    """Preprocess PIL image for the ArcFace model"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Resize to the expected input size
        resized = cv2.resize(img_array, (224, 224))
        
        # Ensure RGB format
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            # PIL images are already in RGB format
            pass
        
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
        return None, "Model not loaded"

    try:
        # Preprocess the image
        preprocessed = preprocess_image(image)
        if preprocessed is None:
            return None, "Error preprocessing image"
        
        # Get embeddings from the model
        embeddings = embedding_model.predict(preprocessed, verbose=0)
        
        # Normalize embeddings
        normalized = embeddings / (np.linalg.norm(embeddings) + 1e-7)
        
        return normalized.flatten().tolist(), "Features extracted successfully"
    except Exception as e:
        return None, f"Error extracting features: {e}"

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

def classify_image(image):
    """Main function for image classification"""
    if model is None:
        return "Model not loaded. Please check if the model file exists."
    
    try:
        # Preprocess the image
        preprocessed = preprocess_image(image)
        if preprocessed is None:
            return "Error preprocessing image"
        
        # Get predictions
        predictions = model.predict(preprocessed, verbose=0)
        
        # Get the predicted class (assuming binary classification: 0=Cat, 1=Dog)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        class_names = ["Cat", "Dog"]
        result = f"Prediction: {class_names[predicted_class]}\nConfidence: {confidence:.4f}"
        
        return result
        
    except Exception as e:
        return f"Error during prediction: {e}"

def compare_images(image1, image2):
    """Compare two images and return similarity score"""
    if image1 is None or image2 is None:
        return "Please upload both images"
    
    # Extract features from both images
    features1, msg1 = extract_features(image1)
    features2, msg2 = extract_features(image2)
    
    if features1 is None or features2 is None:
        return f"Error extracting features: {msg1}, {msg2}"
    
    # Compare features
    similarity = compare_features(features1, features2)
    
    return f"Similarity Score: {similarity:.4f}\n({'High similarity' if similarity > 0.8 else 'Medium similarity' if similarity > 0.5 else 'Low similarity'})"

def extract_features_api(image):
    """API endpoint to extract features from an image"""
    if image is None:
        return {"error": "No image provided"}
    
    try:
        features, message = extract_features(image)
        
        if features is None:
            return {
                "success": False,
                "error": message,
                "features": None
            }
        
        return {
            "success": True,
            "message": message,
            "features": features,
            "feature_length": len(features)
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "features": None
        }

def batch_compare_features(query_features, database_features_list):
    """
    Compare query features with multiple database features
    Returns list of similarities
    """
    try:
        if not query_features or not database_features_list:
            return {"error": "Missing features data"}
        
        similarities = []
        
        for idx, db_features in enumerate(database_features_list):
            if db_features:
                similarity = compare_features(query_features, db_features)
                similarities.append({
                    "index": idx,
                    "similarity": similarity
                })
            else:
                similarities.append({
                    "index": idx,
                    "similarity": 0.0
                })
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            "success": True,
            "similarities": similarities,
            "total_compared": len(similarities)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "similarities": []
        }

# Load models on startup
model_loaded = load_models()

# Create Gradio interface
with gr.Blocks(title="Cat vs Dog Classifier with ArcFace") as demo:
    gr.Markdown("# Cat vs Dog Classifier with ArcFace")
    gr.Markdown("Upload an image to classify it as a cat or dog, extract features, or compare images for similarity.")
    
    with gr.Tab("Image Classification"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")
                classify_btn = gr.Button("Classify Image")
            with gr.Column():
                classification_output = gr.Textbox(label="Classification Result", lines=3)
        
        classify_btn.click(
            fn=classify_image,
            inputs=image_input,
            outputs=classification_output
        )
    
    with gr.Tab("Feature Extraction"):
        gr.Markdown("Extract feature vectors from images for similarity matching")
        with gr.Row():
            with gr.Column():
                feature_image_input = gr.Image(type="pil", label="Upload Image")
                extract_btn = gr.Button("Extract Features")
            with gr.Column():
                feature_output = gr.JSON(label="Feature Extraction Result")
        
        extract_btn.click(
            fn=extract_features_api,
            inputs=feature_image_input,
            outputs=feature_output
        )
    
    with gr.Tab("Image Similarity"):
        with gr.Row():
            with gr.Column():
                image1_input = gr.Image(type="pil", label="Upload First Image")
                image2_input = gr.Image(type="pil", label="Upload Second Image")
                compare_btn = gr.Button("Compare Images")
            with gr.Column():
                similarity_output = gr.Textbox(label="Similarity Result", lines=3)
        
        compare_btn.click(
            fn=compare_images,
            inputs=[image1_input, image2_input],
            outputs=similarity_output
        )
    
    with gr.Tab("Batch Feature Comparison"):
        gr.Markdown("Compare one set of features against multiple database features")
        with gr.Row():
            with gr.Column():
                query_features_input = gr.Textbox(
                    label="Query Features (JSON array)", 
                    placeholder='[0.1, 0.2, 0.3, ...]'
                )
                db_features_input = gr.Textbox(
                    label="Database Features (JSON array of arrays)", 
                    placeholder='[[0.1, 0.2, ...], [0.3, 0.4, ...], ...]'
                )
                batch_compare_btn = gr.Button("Compare Features")
            with gr.Column():
                batch_output = gr.JSON(label="Batch Comparison Result")
        
        def batch_compare_wrapper(query_str, db_str):
            try:
                import json
                query_features = json.loads(query_str) if query_str else None
                db_features = json.loads(db_str) if db_str else None
                return batch_compare_features(query_features, db_features)
            except json.JSONDecodeError as e:
                return {"error": f"Invalid JSON: {str(e)}"}
        
        batch_compare_btn.click(
            fn=batch_compare_wrapper,
            inputs=[query_features_input, db_features_input],
            outputs=batch_output
        )
    
    # Add some example images if you have them
    gr.Examples(
        examples=[
            # Add paths to example images here if you have them
            # ["example_cat.jpg"],
            # ["example_dog.jpg"]
        ],
        inputs=image_input
    )

if __name__ == "__main__":
    demo.launch()
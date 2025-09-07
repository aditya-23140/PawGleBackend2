import io
from PIL import Image
from gradio_client import Client
import logging
import json

logger = logging.getLogger(__name__)

class PawgleAPIClient:
    def __init__(self):
        self.space_url = "nexus-neon/pawgle"
        self._client = None
    
    @property
    def client(self):
        """Lazy load the Gradio client"""
        if self._client is None:
            try:
                # Use the correct URL format for Hugging Face Spaces
                self._client = Client(f"https://huggingface.co/spaces/{self.space_url}")
                logger.info(f"Connected to Pawgle HF Space: {self.space_url} ✔")
            except Exception as e:
                logger.error(f"Failed to connect to HF Space: {e}")
                raise
        return self._client
    
    def extract_features(self, image):
        """
        Extract features from an image using the dedicated feature extraction endpoint
        Returns: (features_list, success_message) or (None, error_message)
        """
        try:
            # Use the provided image path directly if it's a string path
            if not isinstance(image, Image.Image):
                temp_path = image
            else:
                # For PIL images, use tempfile to create a proper temporary file
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_path = temp_file.name
                    image.save(temp_path, format='JPEG')
            
            # Use the new feature extraction endpoint
            result = self.client.predict(
                temp_path,
                api_name="/extract_features_api"  # New endpoint from updated Gradio app
            )
            
            # Parse the JSON result
            if isinstance(result, dict):
                if result.get('success'):
                    return result.get('features'), result.get('message', 'Features extracted successfully')
                else:
                    return None, result.get('error', 'Feature extraction failed')
            else:
                # Handle case where result might be a string
                try:
                    result_dict = json.loads(result)
                    if result_dict.get('success'):
                        return result_dict.get('features'), result_dict.get('message', 'Features extracted successfully')
                    else:
                        return None, result_dict.get('error', 'Feature extraction failed')
                except json.JSONDecodeError:
                    return None, f"Unexpected result format: {result}"
                    
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None, f"Feature extraction failed: {str(e)}"
    
    def classify_pet(self, image):
        """Classify pet using your deployed model"""
        try:
            # Use the provided image path directly if it's a string path
            if not isinstance(image, Image.Image):
                temp_path = image
            else:
                # For PIL images, use tempfile to create a proper temporary file
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_path = temp_file.name
                    image.save(temp_path, format='JPEG')
            
            result = self.client.predict(
                temp_path,
                api_name="/classify_image"
            )
            
            return result, "Classification successful"
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return None, f"Classification failed: {str(e)}"
    
    def compare_images_similarity(self, image1, image2):
        """Compare two images for similarity"""
        try:
            import tempfile
            import os
            
            # Handle first image
            if not isinstance(image1, Image.Image):
                temp_path1 = image1
            else:
                # For PIL images, use tempfile to create a proper temporary file
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_path1 = temp_file.name
                    image1.save(temp_path1, format='JPEG')
            
            # Handle second image
            if not isinstance(image2, Image.Image):
                temp_path2 = image2
            else:
                # For PIL images, use tempfile to create a proper temporary file
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_path2 = temp_file.name
                    image2.save(temp_path2, format='JPEG')
            
            result = self.client.predict(
                temp_path1,
                temp_path2,
                api_name="/compare_images"
            )
            
            # Parse similarity score from result string
            # Your result format: "Similarity Score: 0.8234\nHigh similarity"
            if "Similarity Score:" in result:
                score_line = result.split('\n')[0]
                score = float(score_line.split(': ')[1])
                return score, "Comparison successful"
            else:
                return None, f"Unexpected result format: {result}"
                
        except Exception as e:
            logger.error(f"Comparison error: {e}")
            return None, f"Comparison failed: {str(e)}"

    def batch_compare_features(self, query_features, database_features_list):
        """
        Compare query features against multiple database features
        Returns: (similarities_list, success_message) or (None, error_message)
        """
        try:
            # Convert features to JSON strings for the API
            query_json = json.dumps(query_features)
            db_json = json.dumps(database_features_list)
            
            result = self.client.predict(
                query_json,
                db_json,
                api_name="/batch_compare_features"  # New endpoint from updated Gradio app
            )
            
            # Parse the result
            if isinstance(result, dict):
                if result.get('success'):
                    return result.get('similarities', []), result.get('message', 'Batch comparison successful')
                else:
                    return None, result.get('error', 'Batch comparison failed')
            else:
                # Handle case where result might be a string
                try:
                    result_dict = json.loads(result) if isinstance(result, str) else result
                    if result_dict.get('success'):
                        return result_dict.get('similarities', []), 'Batch comparison successful'
                    else:
                        return None, result_dict.get('error', 'Batch comparison failed')
                except json.JSONDecodeError:
                    return None, f"Unexpected result format: {result}"
                    
        except Exception as e:
            logger.error(f"Batch comparison error: {e}")
            return None, f"Batch comparison failed: {str(e)}"

# Initialize global client
pawgle_client = PawgleAPIClient()
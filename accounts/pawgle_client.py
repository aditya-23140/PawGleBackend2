import io
from PIL import Image
from gradio_client import Client
import logging
import json
import tempfile
import os

logger = logging.getLogger(__name__)

class PawgleAPIClient:
    def __init__(self):
        # Make sure this matches your actual HuggingFace space
        self.space_url = "nexus-neon/pawgle"  # Verify this is correct
        self._client = None
        self.max_retries = 3
    
    @property
    def client(self):
        """Lazy load the Gradio client with retry logic"""
        if self._client is None:
            for attempt in range(self.max_retries):
                try:
                    # Try different connection methods
                    self._client = Client(self.space_url)
                    logger.info(f"✓ Connected to Pawgle HF Space: {self.space_url}")
                    
                    # Test the connection by checking available endpoints
                    try:
                        endpoints = self._client.view_api()
                        logger.info(f"Available endpoints: {endpoints}")
                        return self._client
                    except Exception as e:
                        logger.warning(f"Could not view API endpoints: {e}")
                        return self._client
                        
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed to connect to HF Space: {e}")
                    if attempt == self.max_retries - 1:
                        raise Exception(f"Failed to connect after {self.max_retries} attempts: {e}")
                    
        return self._client
    
    def extract_features(self, image_path_or_pil):
        """
        Extract features from an image using the dedicated feature extraction endpoint
        Returns: (features_list, success_message) or (None, error_message)
        """
        temp_file_path = None
        try:
            logger.info("Starting feature extraction...")
            
            # Handle different input types
            if isinstance(image_path_or_pil, str):
                # It's already a file path
                input_path = image_path_or_pil
                logger.info(f"Using provided file path: {input_path}")
            elif isinstance(image_path_or_pil, Image.Image):
                # Convert PIL Image to temporary file
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    image_path_or_pil.save(temp_file.name, format='JPEG', quality=95)
                    input_path = temp_file.name
                    temp_file_path = temp_file.name
                    logger.info(f"Created temp file: {input_path}")
            else:
                return None, f"Invalid image input type: {type(image_path_or_pil)}"
            
            # Verify file exists and is readable
            if not os.path.exists(input_path):
                return None, f"Image file does not exist: {input_path}"
            
            file_size = os.path.getsize(input_path)
            logger.info(f"Image file size: {file_size} bytes")
            
            if file_size == 0:
                return None, "Image file is empty"
            
            # Call the Gradio interface with proper error handling
            logger.info("Calling HuggingFace Space API...")
            
            try:
                # Make sure this API name matches exactly what's in your app.py
                result = self.client.predict(
                    input_path,
                    api_name="/extract_features_api"  # This must match your Gradio app
                )
                
                logger.info(f"HF Space returned: {type(result)} - {str(result)[:200]}...")
                
            except Exception as api_error:
                logger.error(f"HF Space API call failed: {api_error}")
                return None, f"HF Space API error: {str(api_error)}"
            
            # Clean up temporary file if created
            if temp_file_path:
                try:
                    os.unlink(temp_file_path)
                    logger.info("Cleaned up temp file")
                except:
                    pass
            
            # Parse the result - your app.py returns JSON string
            try:
                if isinstance(result, str):
                    # Try to parse as JSON
                    result_dict = json.loads(result)
                    logger.info(f"Parsed JSON result: {result_dict.keys()}")
                    
                    if result_dict.get('success'):
                        features = result_dict.get('features')
                        if features and isinstance(features, list):
                            logger.info(f"✓ Features extracted successfully: {len(features)} dimensions")
                            return features, result_dict.get('message', 'Features extracted successfully')
                        else:
                            return None, "No features in successful response"
                    else:
                        error_msg = result_dict.get('error', 'Feature extraction failed')
                        logger.error(f"HF Space returned error: {error_msg}")
                        return None, error_msg
                        
                elif isinstance(result, dict):
                    # Direct dict response
                    if result.get('success'):
                        features = result.get('features')
                        if features:
                            return features, result.get('message', 'Features extracted successfully')
                        else:
                            return None, "No features in response"
                    else:
                        return None, result.get('error', 'Feature extraction failed')
                else:
                    logger.error(f"Unexpected result type: {type(result)}")
                    return None, f"Unexpected result format: {result}"
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                return None, f"Invalid JSON response: {result}"
                    
        except Exception as e:
            # Clean up temporary file if created
            if temp_file_path:
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            logger.error(f"Feature extraction error: {e}")
            return None, f"Feature extraction failed: {str(e)}"
    
    def classify_pet(self, image_path_or_pil):
        """Classify pet using your deployed model"""
        temp_file_path = None
        try:
            logger.info("Starting pet classification...")
            
            # Handle different input types
            if isinstance(image_path_or_pil, str):
                input_path = image_path_or_pil
            elif isinstance(image_path_or_pil, Image.Image):
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    image_path_or_pil.save(temp_file.name, format='JPEG', quality=95)
                    input_path = temp_file.name
                    temp_file_path = temp_file.name
            else:
                return None, "Invalid image input type"
            
            # Call classification API
            result = self.client.predict(
                input_path,
                api_name="/classify_image"  # This should match your Gradio app
            )
            
            # Clean up temporary file if created
            if temp_file_path:
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            
            logger.info(f"Classification result: {result}")
            return result, "Classification successful"
            
        except Exception as e:
            # Clean up temporary file if created
            if temp_file_path:
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            logger.error(f"Classification error: {e}")
            return None, f"Classification failed: {str(e)}"
    
    def compare_images_similarity(self, image1, image2):
        """Compare two images for similarity"""
        temp_path1 = None
        temp_path2 = None
        try:
            # Handle first image
            if isinstance(image1, str):
                input_path1 = image1
            elif isinstance(image1, Image.Image):
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    image1.save(temp_file.name, format='JPEG')
                    input_path1 = temp_file.name
                    temp_path1 = temp_file.name
            else:
                return None, "Invalid first image type"
            
            # Handle second image
            if isinstance(image2, str):
                input_path2 = image2
            elif isinstance(image2, Image.Image):
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    image2.save(temp_file.name, format='JPEG')
                    input_path2 = temp_file.name
                    temp_path2 = temp_file.name
            else:
                return None, "Invalid second image type"
            
            result = self.client.predict(
                input_path1,
                input_path2,
                api_name="/compare_images"
            )
            
            # Clean up temporary files
            for temp_path in [temp_path1, temp_path2]:
                if temp_path:
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
            
            # Parse similarity score from result string
            if "Similarity Score:" in result:
                score_line = result.split('\n')[0]
                score = float(score_line.split(': ')[1])
                return score, "Comparison successful"
            else:
                return None, f"Unexpected result format: {result}"
                
        except Exception as e:
            # Clean up temporary files
            for temp_path in [temp_path1, temp_path2]:
                if temp_path:
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
            logger.error(f"Comparison error: {e}")
            return None, f"Comparison failed: {str(e)}"

    def batch_compare_features(self, query_features, database_features_list):
        """
        Compare query features against multiple database features
        Returns: (similarities_list, success_message) or (None, error_message)
        """
        try:
            logger.info(f"Starting batch comparison with {len(database_features_list)} database entries")
            
            # Convert features to JSON strings for the API
            query_json = json.dumps(query_features)
            db_json = json.dumps(database_features_list)
            
            result = self.client.predict(
                query_json,
                db_json,
                api_name="/batch_compare_features"
            )
            
            # Parse the result
            if isinstance(result, dict):
                if result.get('success'):
                    return result.get('similarities', []), result.get('message', 'Batch comparison successful')
                else:
                    return None, result.get('error', 'Batch comparison failed')
            elif isinstance(result, str):
                try:
                    result_dict = json.loads(result)
                    if result_dict.get('success'):
                        return result_dict.get('similarities', []), 'Batch comparison successful'
                    else:
                        return None, result_dict.get('error', 'Batch comparison failed')
                except json.JSONDecodeError:
                    return None, f"Unexpected result format: {result}"
            else:
                return None, f"Unexpected result type: {type(result)}"
                    
        except Exception as e:
            logger.error(f"Batch comparison error: {e}")
            return None, f"Batch comparison failed: {str(e)}"

# Initialize global client
pawgle_client = PawgleAPIClient()
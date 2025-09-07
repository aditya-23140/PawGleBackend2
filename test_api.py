import os
import sys
import django
import time
import logging
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'animal.settings')
django.setup()

# Import the PawgleAPIClient
from accounts.pawgle_client import pawgle_client

# Test image path - update this to a valid image path in your system
test_image_path = 'test/flickr_cat_000076.jpg'

# Check if the file exists
if not os.path.exists(test_image_path):
    logger.error(f"Error: Test image not found at {test_image_path}")
    sys.exit(1)

logger.info(f"Testing feature extraction with image: {test_image_path}")

# Print the current space URL being used
logger.info(f"Using HuggingFace Space URL: {pawgle_client.space_url}")

# Define a class to hold the results
class ResultHolder:
    def __init__(self):
        self.features = None
        self.message = None
        self.error = None
        self.completed = False

# Function to run in a separate thread
def extract_features_thread(result_holder, image_path):
    try:
        logger.info("Starting feature extraction in thread...")
        features, message = pawgle_client.extract_features(image_path)
        result_holder.features = features
        result_holder.message = message
        result_holder.completed = True
        logger.info("Feature extraction thread completed")
    except Exception as e:
        logger.error(f"Error in extraction thread: {str(e)}")
        result_holder.error = str(e)
        result_holder.completed = True

# Create a result holder
result = ResultHolder()

# Create and start the thread
extraction_thread = threading.Thread(target=extract_features_thread, args=(result, test_image_path))
extraction_thread.daemon = True

logger.info("Starting feature extraction thread...")
start_time = time.time()
extraction_thread.start()

# Wait for the thread to complete with a timeout
timeout = 30  # seconds
extraction_thread.join(timeout)

elapsed_time = time.time() - start_time
logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")

if not result.completed:
    logger.error(f"API call timed out after {timeout} seconds")
else:
    if result.error:
        logger.error(f"Error during feature extraction: {result.error}")
    elif result.features and len(result.features) > 0:
        logger.info(f"✅ Success! Features extracted: {len(result.features)} dimensions")
        logger.info(f"First few feature values: {result.features[:5]}")
    else:
        logger.error(f"❌ Failed to extract features: {result.message}")
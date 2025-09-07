import os
import uuid
from django.conf import settings
from supabase import create_client, Client
from django.core.files.storage import Storage
from django.core.files.base import ContentFile
from django.utils.deconstruct import deconstructible
import mimetypes

@deconstructible
class SupabaseStorage(Storage):
    def __init__(self, bucket_name=None):
        self.bucket_name = bucket_name or settings.SUPABASE_BUCKET_NAME
        # Use the service role key for server-side operations
        supabase_key = getattr(settings, 'SUPABASE_SERVICE_KEY', settings.SUPABASE_KEY)
        self.supabase: Client = create_client(
            settings.SUPABASE_URL,
            supabase_key
        )

    def _save(self, name, content):
        # Generate unique filename with proper path structure
        file_extension = os.path.splitext(name)[1].lower()
        unique_name = f"pets/{uuid.uuid4()}{file_extension}"
        
        # Read file content
        content.seek(0)
        file_content = content.read()
        
        # Get content type
        content_type, _ = mimetypes.guess_type(name)
        if not content_type:
            # Set appropriate content type based on file extension
            if file_extension in ['.jpg', '.jpeg']:
                content_type = 'image/jpeg'
            elif file_extension == '.png':
                content_type = 'image/png'
            elif file_extension == '.gif':
                content_type = 'image/gif'
            elif file_extension == '.webp':
                content_type = 'image/webp'
            else:
                content_type = 'application/octet-stream'

        try:
            # Upload to Supabase Storage with proper options
            response = self.supabase.storage.from_(self.bucket_name).upload(
                path=unique_name,
                file=file_content,
                file_options={
                    "content-type": content_type,
                    "cache-control": "3600"
                }
            )
            
            # Check response format - Supabase Python client returns different response formats
            if hasattr(response, 'data') and response.data:
                return unique_name
            elif isinstance(response, dict):
                if 'error' in response:
                    raise Exception(f"Upload failed: {response['error']}")
                elif 'path' in response or 'Key' in response:
                    return unique_name
            
            # If we get here, assume success if no error was raised
            return unique_name
                
        except Exception as e:
            # Log the full error for debugging
            print(f"Supabase upload error: {str(e)}")
            raise Exception(f"Failed to upload to Supabase: {str(e)}")

    def delete(self, name):
        try:
            response = self.supabase.storage.from_(self.bucket_name).remove([name])
            if hasattr(response, 'error') and response.error:
                print(f"Error deleting {name} from Supabase: {response.error}")
        except Exception as e:
            # Log the error but don't raise it to avoid breaking the model deletion
            print(f"Failed to delete {name} from Supabase: {str(e)}")

    def exists(self, name):
        try:
            # Try to get the file info to check if it exists
            response = self.supabase.storage.from_(self.bucket_name).list(
                path=os.path.dirname(name) if '/' in name else "",
                limit=100
            )
            
            if hasattr(response, 'data') and isinstance(response.data, list):
                filename = os.path.basename(name)
                return any(item.get('name') == filename for item in response.data)
            elif isinstance(response, list):
                filename = os.path.basename(name)
                return any(item.get('name') == filename for item in response)
            return False
        except Exception as e:
            print(f"Error checking if {name} exists: {str(e)}")
            return False

    def url(self, name):
        try:
            # Get public URL for the file using the newer API
            response = self.supabase.storage.from_(self.bucket_name).get_public_url(name)
            
            # Handle different response formats
            if isinstance(response, str):
                return response
            elif hasattr(response, 'publicURL'):
                return response.publicURL
            elif isinstance(response, dict) and 'publicURL' in response:
                return response['publicURL']
            else:
                # Fallback to constructing URL manually
                return f"{settings.SUPABASE_URL}/storage/v1/object/public/{self.bucket_name}/{name}"
                
        except Exception as e:
            print(f"Failed to get URL for {name}: {str(e)}")
            # Return fallback URL
            return f"{settings.SUPABASE_URL}/storage/v1/object/public/{self.bucket_name}/{name}"

    def size(self, name):
        try:
            # Get file info to retrieve size
            response = self.supabase.storage.from_(self.bucket_name).list(
                path=os.path.dirname(name) if '/' in name else ""
            )
            
            if hasattr(response, 'data') and isinstance(response.data, list):
                filename = os.path.basename(name)
                for item in response.data:
                    if item.get('name') == filename:
                        return item.get('metadata', {}).get('size', 0)
            return 0
        except Exception as e:
            print(f"Error getting size for {name}: {str(e)}")
            return 0

    def get_accessed_time(self, name):
        return None

    def get_created_time(self, name):
        return None

    def get_modified_time(self, name):
        return None

    def listdir(self, path):
        """List the contents of the specified path."""
        try:
            response = self.supabase.storage.from_(self.bucket_name).list(path=path)
            
            directories = []
            files = []
            
            data = response.data if hasattr(response, 'data') else response
            
            if isinstance(data, list):
                for item in data:
                    if item.get('metadata') and item['metadata'].get('mimetype'):
                        files.append(item['name'])
                    else:
                        directories.append(item['name'])
            
            return directories, files
        except Exception as e:
            print(f"Error listing directory {path}: {str(e)}")
            return [], []
            
    def _open(self, name, mode='rb'):
        """Open the specified file from storage."""
        try:
            # Download the file from Supabase
            response = self.supabase.storage.from_(self.bucket_name).download(name)
            
            # Create a ContentFile from the downloaded data
            content = ContentFile(response)
            content.name = name
            return content
        except Exception as e:
            print(f"Error opening file {name}: {str(e)}")
            raise
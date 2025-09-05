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
        self.supabase: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_KEY
        )

    def _save(self, name, content):
        # Generate unique filename
        file_extension = os.path.splitext(name)[1]
        unique_name = f"{uuid.uuid4()}{file_extension}"
        
        # Read file content
        content.seek(0)
        file_content = content.read()
        
        # Get content type
        content_type, _ = mimetypes.guess_type(name)
        if not content_type:
            content_type = 'application/octet-stream'

        try:
            # Upload to Supabase Storage
            response = self.supabase.storage.from_(self.bucket_name).upload(
                unique_name,
                file_content,
                file_options={
                    "content-type": content_type
                }
            )
            
            # Check if upload was successful
            if hasattr(response, 'data') and response.data is not None:
                return unique_name
            else:
                raise Exception(f"Upload failed: {response}")
                
        except Exception as e:
            raise Exception(f"Failed to upload to Supabase: {str(e)}")

    def delete(self, name):
        try:
            self.supabase.storage.from_(self.bucket_name).remove([name])
        except Exception as e:
            # Log the error but don't raise it to avoid breaking the model deletion
            print(f"Failed to delete {name} from Supabase: {str(e)}")

    def exists(self, name):
        try:
            response = self.supabase.storage.from_(self.bucket_name).list(
                path="",
                limit=1,
                search=name
            )
            return len(response) > 0
        except:
            return False

    def url(self, name):
        try:
            # Get public URL for the file
            response = self.supabase.storage.from_(self.bucket_name).get_public_url(name)
            return response
        except Exception as e:
            print(f"Failed to get URL for {name}: {str(e)}")
            return ""

    def size(self, name):
        # This is optional, return 0 if not implemented
        return 0

    def get_accessed_time(self, name):
        # Required method for Django Storage
        return None

    def get_created_time(self, name):
        # Required method for Django Storage
        return None

    def get_modified_time(self, name):
        # Required method for Django Storage
        return None

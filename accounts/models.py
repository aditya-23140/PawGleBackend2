from django.db import models
from django.conf import settings
from django.utils import timezone
from django.core.validators import MinLengthValidator, RegexValidator, MinValueValidator, MaxValueValidator
import json
import random
import uuid
from .storage import SupabaseStorage

# Define validation functions outside the model
def validate_json_dict(value):
    """Ensure value is a JSON-serializable dictionary"""
    if not isinstance(value, dict):
        raise ValueError("Must be a dictionary")
    json.dumps(value)  # Test JSON serialization

def validate_json_list(value):
    """Ensure value is a JSON-serializable list"""
    if not isinstance(value, list):
        raise ValueError("Must be a list")
    json.dumps(value)  # Test JSON serialization

class Pet(models.Model):
    CATEGORY_CHOICES = [
        ('Domestic', 'Domestic'),
        ('Wild', 'Wild'),
        ('Poultry', 'Poultry'),
        ('Livestock', 'Livestock')
    ]

    # Primary key using random number
    id = models.IntegerField(primary_key=True, editable=False)
    
    # Required fields with validation
    name = models.CharField(
        max_length=100,
        validators=[
            MinLengthValidator(2),
            RegexValidator(
                regex=r'^[a-zA-Z0-9 \'-]+$',
                message='Name can only contain letters, numbers, spaces, apostrophes, and hyphens'
            )
        ]
    )
    type = models.CharField(
        max_length=100,
        validators=[
            MinLengthValidator(2),
            RegexValidator(
                regex=r'^[a-zA-Z ]+$',
                message='Type can only contain letters and spaces'
            )
        ]
    )
    category = models.CharField(
        max_length=100,
        choices=CATEGORY_CHOICES
    )
    breed = models.CharField(
        max_length=100,
        validators=[
            MinLengthValidator(2),
            RegexValidator(
                regex=r'^[a-zA-Z ]+$',
                message='Breed can only contain letters and spaces'
            )
        ]
    )
    
    # JSON fields with validation
    additionalInfo = models.JSONField(
        default=dict,
        validators=[validate_json_dict]
    )
    images = models.JSONField(
        default=list,
        validators=[validate_json_list]
    )
    features = models.JSONField(
        default=list,
        validators=[validate_json_list]
    )
    
    # System-managed fields
    isPublic = models.BooleanField(default=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='pets'
    )
    
    # Using UUID-based animal_id to guarantee uniqueness
    animal_id = models.CharField(
        max_length=40,
        unique=True,
        editable=False
    )
    
    registered_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.animal_id})"

    def generate_unique_id(self):
        """Generate a unique random ID between 100000 and 999999"""
        while True:
            unique_id = random.randint(100000, 999999)
            if not Pet.objects.filter(id=unique_id).exists():
                return unique_id

    def save(self, *args, **kwargs):
        # Generate random ID if not set
        if not self.id:
            self.id = self.generate_unique_id()
            
        # Generate completely unique animal_id if not set
        if not self.animal_id:
            # Use UUID to guarantee uniqueness, but format it to look like ANIxxxx
            # Take first 4 hex chars from a UUID and format
            uid = str(uuid.uuid4())[:4].upper()
            self.animal_id = f"ANI{uid}"
            
            # Verify it's unique (extremely unlikely to have a collision, but just in case)
            while Pet.objects.filter(animal_id=self.animal_id).exists():
                uid = str(uuid.uuid4())[:4].upper()
                self.animal_id = f"ANI{uid}"
        
        # Type enforcement
        if not isinstance(self.additionalInfo, dict):
            self.additionalInfo = {}
        if not isinstance(self.images, list):
            self.images = []
        if not isinstance(self.features, list):
            self.features = []
            
        super().save(*args, **kwargs)

    class Meta:
        indexes = [
            models.Index(fields=['animal_id']),
            models.Index(fields=['owner']),
        ]
        ordering = ['-registered_at']

class PetLocation(models.Model):
    STATUS_CHOICES = [
        ('lost', 'Lost'),
        ('found', 'Found'),
        ('resolved', 'Resolved'),
    ]
    
    # Instead of a direct foreign key, make the pet reference optional
    pet = models.ForeignKey('Pet', on_delete=models.SET_NULL, related_name='locations', 
                           null=True, blank=True)
    
    # Add fields to describe the pet when there's no reference
    pet_name = models.CharField(max_length=100, blank=True)
    pet_type = models.CharField(max_length=100, blank=True)
    pet_breed = models.CharField(max_length=100, blank=True)
    pet_description = models.TextField(blank=True)
    
    # Location data
    latitude = models.FloatField(
        validators=[
            MinValueValidator(-90.0),
            MaxValueValidator(90.0)
        ]
    )
    longitude = models.FloatField(
        validators=[
            MinValueValidator(-180.0),
            MaxValueValidator(180.0)
        ]
    )
    
    # Status fields
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='lost')
    description = models.TextField(blank=True)
    reported_at = models.DateTimeField(auto_now_add=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    
    # Contact information
    contact_name = models.CharField(max_length=100, blank=True)
    contact_phone = models.CharField(max_length=20, blank=True)
    contact_email = models.EmailField(blank=True)
    
    # Additional information
    last_seen_date = models.DateField(null=True, blank=True)
    last_seen_time = models.TimeField(null=True, blank=True)
    
    # Add image field
    image = models.ImageField(
        storage=SupabaseStorage(),
        null=True,
        blank=True
    )
    
    # Add features field for image recognition
    features = models.JSONField(default=list, validators=[validate_json_list])
    
    # Add field to track if this is a user's current location
    is_user_location = models.BooleanField(default=False)
    
    class Meta:
        indexes = [
            models.Index(fields=['status']),
            models.Index(fields=['pet']),
            models.Index(fields=['reported_at']),
        ]
        ordering = ['-reported_at']
    
    def __str__(self):
        if self.pet:
            pet_name = self.pet.name
        else:
            pet_name = self.pet_name or "Unknown Pet"
        return f"{pet_name} - {self.get_status_display()} at {self.reported_at.strftime('%Y-%m-%d %H:%M')}"
    
    def mark_as_found(self):
        """Mark a lost pet as found and update the resolved_at timestamp"""
        if self.status == 'lost':
            self.status = 'resolved'
            self.resolved_at = timezone.now()
            self.save()
    
    def mark_as_lost(self):
        """Mark a pet as lost"""
        self.status = 'lost'
        self.resolved_at = None
        self.save()
    
    def link_to_pet(self, pet):
        """Link this location record to a registered pet"""
        self.pet = pet
        self.save()
    
    def extract_and_store_features(self):
        """Extract features from the image and store them"""
        if not self.image:
            return False
            
        try:
            import cv2
            from your_feature_extraction_module import extract_features
            
            image = cv2.imread(self.image.path)
            if image is None:
                return False
                
            features = extract_features(image)
            if features:
                self.features = features
                self.save(update_fields=['features'])
                return True
            return False
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return False
    
    def to_map_marker(self, request=None):
        """Convert to a format suitable for map markers"""
        marker = {
            'id': self.id,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'status': self.status,
            'animal_name': self.pet.name if self.pet else self.pet_name,
            'type': self.pet.type if self.pet else self.pet_type,
            'breed': self.pet.breed if self.pet else self.pet_breed,
            'isUserLocation': self.is_user_location,
        }
        
        # Add image URL if available and request is provided
        if self.image and request:
            marker['image_url'] = request.build_absolute_uri(self.image.url)
        
        return marker
    
    def save(self, *args, **kwargs):
        # Call the original save method
        super().save(*args, **kwargs)
        
        # Extract features if image exists and features are empty
        if self.image and not self.features:
            self.extract_and_store_features()

class Notification(models.Model):
    recipient = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE
    )
    verb = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    target = models.ForeignKey(Pet, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

class Conversation(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    pet_location = models.ForeignKey(PetLocation, on_delete=models.CASCADE, related_name='conversations')
    reporter_email = models.EmailField()
    reporter_name = models.CharField(max_length=100)
    owner_share_info = models.BooleanField(default=False)
    reporter_share_info = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        if self.pet_location.pet:
            pet_name = self.pet_location.pet.name
        else:
            pet_name = self.pet_location.pet_name or "Unknown Pet"
        return f"Conversation {self.id} - {pet_name}"


from django.db import models
from django.conf import settings

class EditedPetImage(models.Model):
    edited_image = models.ImageField(
        storage=SupabaseStorage(),
        help_text="The edited image file"
    )
    edit_metadata = models.JSONField(
        default=dict,
        help_text="JSON store of editing parameters and history"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='edited_pet_images'
    )

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['owner']),
            models.Index(fields=['created_at']),
        ]
        verbose_name = "Edited Pet Image"
        verbose_name_plural = "Edited Pet Images"

    def __str__(self):
        return f"Edited image by {self.owner.username} at {self.created_at}"

from django.contrib.auth.models import User
from rest_framework import serializers
from .models import Pet, PetLocation,Notification
from random import randint

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'email')

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    confirm_password = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'password', 'confirm_password')

    def validate(self, data):
        if data['password'] != data['confirm_password']:
            raise serializers.ValidationError("Passwords don't match.")
        return data

    def create(self, validated_data):
        user = User.objects.create(
            username=validated_data['username'],
            email=validated_data['email']
        )
        user.set_password(validated_data['password'])
        user.save()
        return user

class PetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Pet
        fields = [
            'id', 'name', 'type', 'category', 'breed', 'isPublic', 
            'additionalInfo', 'animal_id', 'registered_at', 
            'images', 'features', 'owner'
        ]
        read_only_fields = ['animal_id', 'registered_at', 'owner']

    def create(self, validated_data):
        # Set the owner to the current user
        validated_data['owner'] = self.context['request'].user
        # Generate a unique animal_id
        validated_data['animal_id'] = f"ANI{randint(100000, 999999)}"
        return super().create(validated_data)

    def update(self, instance, validated_data):
        # Update the Pet instance
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()
        return instance

class PetLocationSerializer(serializers.ModelSerializer):
    animal_name = serializers.CharField(source='pet.name', read_only=True)
    animal_id = serializers.CharField(source='pet.animal_id', read_only=True)
    type = serializers.CharField(source='pet.type', read_only=True)
    breed = serializers.CharField(source='pet.breed', read_only=True)
    category = serializers.CharField(source='pet.category', read_only=True)
    owner_name = serializers.CharField(source='pet.owner.username', read_only=True)
    image_url = serializers.SerializerMethodField()
    
    class Meta:
        model = PetLocation
        fields = [
            'id', 'pet', 'latitude', 'longitude', 'status', 'description',
            'reported_at', 'resolved_at', 'contact_name', 'contact_phone',
            'contact_email', 'last_seen_date', 'last_seen_time',
            'animal_name', 'animal_id', 'type', 'breed', 'category', 
            'owner_name', 'image', 'image_url'
        ]
        read_only_fields = ['reported_at', 'resolved_at']
    
    def get_image_url(self, obj):
        if obj.image:
            return obj.image.url  # This will now return the Supabase URL
        return None

class ReportPetLocationSerializer(serializers.Serializer):
    # Pet information
    animal_name = serializers.CharField(required=True)
    type = serializers.CharField(required=True)
    breed = serializers.CharField(required=True)
    category = serializers.ChoiceField(choices=Pet.CATEGORY_CHOICES, required=True)
    
    # Register pet option
    register_pet = serializers.BooleanField(default=False)
    
    # Location information
    status = serializers.ChoiceField(choices=['lost', 'found'], required=True)
    description = serializers.CharField(required=False, allow_blank=True)
    latitude = serializers.FloatField(required=True)
    longitude = serializers.FloatField(required=True)
    contact_name = serializers.CharField(required=False, allow_blank=True)
    contact_phone = serializers.CharField(required=False, allow_blank=True)
    contact_email = serializers.EmailField(required=False, allow_blank=True)
    last_seen_date = serializers.DateField(required=False, allow_null=True)
    last_seen_time = serializers.TimeField(required=False, allow_null=True)
    image = serializers.ImageField(required=False)
    
    def create(self, validated_data):
        user = self.context['request'].user
        register_pet = validated_data.pop('register_pet', False)
        
        # Pet information
        animal_name = validated_data.pop('animal_name')
        animal_type = validated_data.pop('type')
        animal_breed = validated_data.pop('breed')
        animal_category = validated_data.pop('category')
        
        pet = None
        
        # If user wants to register the pet or it's their own lost pet
        if register_pet or validated_data['status'] == 'lost':
            try:
                # Try to find an existing pet owned by this user
                pet = Pet.objects.get(
                    owner=user,
                    name=animal_name,
                    type=animal_type,
                    breed=animal_breed
                )
                # Update category if it's different
                if pet.category != animal_category:
                    pet.category = animal_category
                    pet.save()
            except Pet.DoesNotExist:
                # Create a new pet
                pet = Pet.objects.create(
                    owner=user,
                    name=animal_name,
                    type=animal_type,
                    breed=animal_breed,
                    category=animal_category
                )
        
        # Create pet location report
        pet_location = PetLocation.objects.create(
            pet=pet,  # This can be None if pet not registered
            pet_name=animal_name,
            pet_type=animal_type,
            pet_breed=animal_breed,
            pet_description=validated_data.get('description', ''),
            latitude=validated_data['latitude'],
            longitude=validated_data['longitude'],
            status=validated_data['status'],
            description=validated_data.get('description', ''),
            contact_name=validated_data.get('contact_name', ''),
            contact_phone=validated_data.get('contact_phone', ''),
            contact_email=validated_data.get('contact_email', ''),
            last_seen_date=validated_data.get('last_seen_date'),
            last_seen_time=validated_data.get('last_seen_time'),
            image=validated_data.get('image')
        )
        
        # Extract features if image is provided
        if validated_data.get('image'):
            success = pet_location.extract_and_store_features()
            if not success:
                print(f"Warning: Could not extract features for pet location {pet_location.id}")
        
        # Create notification if this is a lost pet report
        if pet and validated_data['status'] == 'lost':
            Notification.objects.create(
                recipient=user,
                verb="Pet marked as lost",
                description=f"You've reported {pet.name} as lost.",
                target=pet
            )
        
        return pet_location

from rest_framework import serializers
from .models import EditedPetImage

class EditedPetImageSerializer(serializers.ModelSerializer):
    edited_image_url = serializers.SerializerMethodField()
    owner = serializers.ReadOnlyField(source='owner.username')

    class Meta:
        model = EditedPetImage
        fields = [
            'id', 'edited_image', 'edit_metadata', 'created_at', 'owner', 'edited_image_url'
        ]
        read_only_fields = ['created_at', 'owner']
        extra_kwargs = {
            'edited_image': {'write_only': True}
        }

    def get_edited_image_url(self, obj):
        if obj.edited_image:
            return obj.edited_image.url  # This will now return the Supabase URL
        return None

    def create(self, validated_data):
        request = self.context.get('request')
        return EditedPetImage.objects.create(
            owner=request.user,
            edited_image=request.FILES.get('edited_image'),
            edit_metadata=validated_data.get('edit_metadata', {})
        )

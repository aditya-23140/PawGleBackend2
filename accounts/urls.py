from django.urls import path
from django.contrib import admin
from .views import (
    ListPetLocationsView, ListLostPetsView, ListFoundPetsView,
    ReportPetLocationView, MarkPetStatusView, UserPetLocationsView,
    # Include your existing views here
    RegisterView, LoginView, ProfileView, AddPetView, PublicPetDashboardView, EditedPetImageViewSet,
    DeletePetView, SearchPetView, EditPetView, GetPetCountView, GetUserCountView, contact_pet_owner,toggle_share_contact_info,share_contact
)

edited_pet_image_list = EditedPetImageViewSet.as_view({
    'get': 'list',
    'post': 'create'
})

edited_pet_image_detail = EditedPetImageViewSet.as_view({
    'get': 'retrieve',
    'put': 'update',
    'patch': 'partial_update',
    'delete': 'destroy'
})

urlpatterns = [
    # Your existing URLs
    
    path("admin/", admin.site.urls),
    path('signup/', RegisterView.as_view(), name='signup'),
    path('login/', LoginView.as_view(), name='login'),
    path('profile/', ProfileView.as_view(), name='profile'),
    path('pets/add/', AddPetView.as_view(), name='add_pet'), 
    path('pets/search/', SearchPetView.as_view(), name='search_pet'), 
    path('dashboard/pets/', PublicPetDashboardView.as_view(), name='public_pet_dashboard'),
    path('pets/<int:pet_id>/delete/', DeletePetView.as_view(), name='delete_pet'),
    path('pets/<int:pet_id>/edit/', EditPetView.as_view(), name='edit_pet'),
    path('pets/count/', GetPetCountView.as_view(), name='get_pet_count'),  
    path('users/count/', GetUserCountView.as_view(), name='get_user_count'),
    
    # New pet location URLs
    path('pets/locations/', ListPetLocationsView.as_view(), name='pet_locations'),
    path('pets/lost/locations/', ListLostPetsView.as_view(), name='lost_pets'),
    path('pets/found/locations/', ListFoundPetsView.as_view(), name='found_pets'),
    path('pets/report/', ReportPetLocationView.as_view(), name='report_pet'),
    path('pets/locations/<int:location_id>/status/', MarkPetStatusView.as_view(), name='update_pet_status'),
    path('user/pet-locations/', UserPetLocationsView.as_view(), name='user_pet_locations'),
    path('pets/contact-owner/', contact_pet_owner, name='contact_pet_owner'),
    path('api/auth/conversations/share-info/', toggle_share_contact_info, name='toggle_share_contact_info'),
    path('share-contact/<uuid:conversation_id>/<str:user_type>/<str:decision>/', share_contact, name='share_contact'),   
    path('edited-pet-images/', edited_pet_image_list, name='edited_pet_image_list'),
    path('edited-pet-images/<int:pk>/', edited_pet_image_detail, name='edited_pet_image_detail'),
]

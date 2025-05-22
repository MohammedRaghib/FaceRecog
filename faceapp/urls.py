from django.urls import path
from . import views

urlpatterns = [
    path("register_worker/", views.worker_registration, name="register"),
    path("check_face/", views.post_face_to_compare, name="check_face"),
]

from django.urls import path
from . import views

urlpatterns = [
    path("login/", views.user_login, name="login"),
    path("register_worker/", views.worker_registration, name="register"),
    path("check_face/", views.post_face_to_compare, name="check_face"),
    path("tasks/<int:personId>/", views.get_worker_task, name="get_worker_task"),
    path("attendance/", views.create_attendance, name="create_attendance"),
]

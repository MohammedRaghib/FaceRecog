from django.urls import path
from .views import FaceVerifyView, FaceRegisterView, FaceIdentifyView, FaceMultiIdentifyView

urlpatterns = [
    path("face/verify/", FaceVerifyView.as_view()),
    path("face/register/", FaceRegisterView.as_view()),
    path("face/identify/", FaceIdentifyView.as_view()),
    path("face/check-multi/", FaceMultiIdentifyView.as_view()),
]

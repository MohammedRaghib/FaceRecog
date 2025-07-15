from django.db import models
from django.utils import timezone

class Worker(models.Model):
    person_id = models.CharField(null=True, blank=True) 
    name = models.CharField(max_length=100)
    face_encoding = models.JSONField()
    role = models.CharField(max_length=100)

    def __str__(self):
        return self.name

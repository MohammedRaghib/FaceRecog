from django.db import models
from django.utils import timezone

class Worker(models.Model):
    person_id = models.CharField(null=True, blank=True) 
    name = models.CharField(max_length=100)
    face_encoding = models.JSONField()
    project_id = models.CharField(max_length=100, null=True, blank=True)

    def __str__(self):
        return self.name

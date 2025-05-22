from django.db import models

class Worker(models.Model):
    name = models.CharField(max_length=100)
    face_encoding = models.JSONField()

    def __str__(self):
        return self.name

from django.db import models
from django.utils import timezone

class Worker(models.Model):
    person_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    face_encoding = models.JSONField()
    role = models.CharField(max_length=100)

    def __str__(self):
        return self.name
    
class Equipment(models.Model): 
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name

class Tasks(models.Model):
    name = models.CharField(max_length=100)
    equipment = models.ManyToManyField(Equipment, related_name='equipment', blank=True)
    labourer = models.ManyToManyField(Worker, related_name='tasks', blank=True)

    def __str__(self):
        return self.name
    
class Attendance(models.Model):
    attendance_id = models.AutoField(primary_key=True)
    attendance_timestamp = models.DateTimeField(default=timezone.now)
    attendance_location = models.CharField(max_length=255)
    attendance_subject = models.CharField(max_length=255, blank=True, null=True)
    # attendance_photo = models.ImageField(upload_to='attendance_photos/', blank=True, null=True)
    attendance_monitor = models.CharField(max_length=255, blank=True, null=True)
    
    # Boolean fields
    attendance_is_check_in = models.BooleanField(default=False)
    attendance_is_approved_by_supervisor = models.BooleanField(default=False)
    attendance_is_entry_permitted = models.BooleanField(default=False)
    attendance_is_work_completed = models.BooleanField(default=False)
    attendance_is_equipment_returned = models.BooleanField(default=False)
    attendance_is_overtime_required = models.BooleanField(default=False)
    attendance_is_overtime_approved = models.BooleanField(default=False)
    attendance_is_incomplete_checkout = models.BooleanField(default=False)
    attendance_is_forced_check_out = models.BooleanField(default=False)
    attendance_is_supervisor_checkout = models.BooleanField(default=False)
    attendance_has_been_checkout_by_supervisor = models.BooleanField(default=False)
    attendance_is_supervisor_checkin = models.BooleanField(default=False)
    attendance_is_unauthorized_entry = models.BooleanField(default=False)
    
    # Other fields
    attendance_access_token = models.CharField(max_length=255, blank=True, null=True)
    attendance_duration_since_supervisor_checkout = models.DurationField(blank=True, null=True)

    def __str__(self):
        return f"Attendance {self.attendance_id} - {self.attendance_timestamp}"

    class Meta:
        verbose_name = "Attendance Record"
        verbose_name_plural = "Attendance Records"
        ordering = ['-attendance_timestamp']
from django.contrib import admin
from .models import Worker, Tasks, Equipment, Attendance
# Register your models here.

# admin.site.register(CustomUser)
admin.site.register(Worker)
admin.site.register(Tasks)
admin.site.register(Equipment)
admin.site.register(Attendance)
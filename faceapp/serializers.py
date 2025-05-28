from rest_framework import serializers
from .models import Worker, Tasks, Equipment

class WorkerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Worker
        fields = ['person_id', 'name']
        
class EquipmentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Equipment
        fields = ['id', 'name']

class TaskSerializer(serializers.ModelSerializer):
    equipment = EquipmentSerializer(many=True)
    class Meta:
        model = Tasks
        fields = ['id', 'name', 'equipment']
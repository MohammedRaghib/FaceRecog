from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.shortcuts import render, redirect
# from django.contrib import messages
# from django.views.decorators.csrf import csrf_exempt

from facenet_pytorch import InceptionResnetV1, MTCNN
from django.http import JsonResponse
from django.contrib.auth import login, authenticate
from rest_framework import status
from PIL import Image
from .models import Worker, Tasks, Equipment, Attendance
from .serializers import WorkerSerializer, TaskSerializer
import numpy as np
import faiss
import os

facenet = InceptionResnetV1(pretrained="vggface2").eval()
mtcnn = MTCNN()

index = None

def update_faiss_index():
    """ Refresh FAISS index safely with stored workers dynamically """
    global index

    workers = Worker.objects.all()

    if not workers.exists():
        index = None
        return

    try:
        worker_embeddings = [np.array(w.face_encoding) for w in workers if w.face_encoding is not None]
        
        if len(worker_embeddings) == 0:
            index = None
            return

        worker_embeddings = np.vstack(worker_embeddings).astype("float32")  
        
        index = faiss.IndexFlatL2(worker_embeddings.shape[1])  
        index.add(worker_embeddings)

        print(f"FAISS index updated with {worker_embeddings.shape[0]} workers.")
    
    except Exception as e:
        print(f"Error updating FAISS index: {e}")
        index = None

update_faiss_index()

@api_view(["POST"])
def worker_registration(request):
    if request.method == "POST":
        name = request.POST.get("name")
        image = request.FILES.get("image")

        if not name or not image:
            return JsonResponse({"message": "Name and image are required.", "success": False}, status=status.HTTP_400_BAD_REQUEST)

        img = Image.open(image).convert("RGB")
        face = mtcnn(img)
        if face is None:
            return JsonResponse({"message": "No face detected.", "success": False}, status=status.HTTP_404_NOT_FOUND)

        embedding = facenet(face.unsqueeze(0)).detach().numpy().tolist()

        Worker.objects.create(name=name, face_encoding=embedding)

        update_faiss_index()
        return JsonResponse({"message": f"Worker {name} registered successfully!", "success": True}, status=status.HTTP_200_OK)

# @csrf_exempt
@api_view(["POST"])
def post_face_to_compare(request):
    if request.method == "POST":
        img = request.FILES.get("image")
        if not img:
            return JsonResponse({"message": "Image is required.", "matchFound": False, "success": False}, status=status.HTTP_400_BAD_REQUEST)

        img = Image.open(img).convert("RGB")
        face = mtcnn(img)
        if face is None:
            return JsonResponse({"message": "No face detected.", "matchFound": False, "success": False}, status=status.HTTP_404_NOT_FOUND)

        uploaded_embedding = facenet(face.unsqueeze(0)).detach().numpy()[0] 
        uploaded_embedding /= np.linalg.norm(uploaded_embedding)

        workers = Worker.objects.all()
        if not workers:
            return JsonResponse({"message": "No registered workers available for comparison.", "matchFound": False, "success": False}, status=status.HTTP_404_NOT_FOUND)

        best_distance = float("inf")
        best_worker = None
        threshold = 0.75

        for worker in workers:
            try:
                worker_embedding = np.array(worker.face_encoding)
                worker_embedding /= np.linalg.norm(worker_embedding)
                distance = np.linalg.norm(uploaded_embedding - worker_embedding)
            
                if distance < best_distance:
                    best_distance = distance
                    best_worker = worker
            except Exception as e:
                print(f"Error processing worker {worker.name}: {e}")
                continue
        
        print(best_distance)
        print(best_worker)
        update_faiss_index()

        if best_worker and best_distance < threshold:
            return JsonResponse({"matched_worker": WorkerSerializer(best_worker).data,"distance": round(best_distance, 4), "matchFound": True, "success": True}, status=status.HTTP_200_OK)

        else:
            return JsonResponse({"message": "No match found.", "matchFound": False, "success": False}, status=status.HTTP_404_NOT_FOUND)
        

@api_view(["GET"])
def get_worker_task(request, personId):
    if request.method == "GET":
        if not personId:
            return JsonResponse({"message": "Worker ID is required.", "success": False}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            worker = Worker.objects.get(person_id=personId)
            tasks = Tasks.objects.filter(labourer=worker)
            serialized_tasks = TaskSerializer(tasks, many=True).data
            
            return JsonResponse({"tasks": serialized_tasks, "success": True}, status=status.HTTP_200_OK)
        except Worker.DoesNotExist:
            return JsonResponse({"message": "Worker not found.", "success": False}, status=status.HTTP_404_NOT_FOUND)

@api_view(["POST"])
def create_attendance(request):
    if request.method == "POST":
        try:
            data = request.POST

            attendance = Attendance(
                attendance_location=data.get('attendance_location'),
                attendance_subject=data.get('attendance_subject_id'),
                attendance_monitor=data.get('attendance_monitor_id'),
                
                attendance_is_check_in=data.get('attendance_is_check_in', False),
                attendance_is_approved_by_supervisor=data.get('attendance_is_approved_by_supervisor', False),
                attendance_is_entry_permitted=data.get('attendance_is_entry_permitted', False),
                attendance_is_work_completed=data.get('attendance_is_work_completed', False),
                attendance_is_equipment_returned=data.get('attendance_is_equipment_returned', False),
                attendance_is_overtime_required=data.get('attendance_is_overtime_required', False),
                attendance_is_overtime_approved=data.get('attendance_is_overtime_approved', False),
                attendance_is_incomplete_checkout=data.get('attendance_is_incomplete_checkout', False),
                attendance_is_forced_check_out=data.get('attendance_is_forced_check_out', False),
                attendance_is_supervisor_checkout=data.get('attendance_is_supervisor_checkout', False),
                attendance_has_been_checkout_by_supervisor=data.get('attendance_has_been_checkout_by_supervisor', False),
                attendance_is_supervisor_checkin=data.get('attendance_is_supervisor_checkin', False),
                attendance_is_unauthorized_entry=data.get('attendance_is_unauthorized_entry', False),
                
                attendance_access_token=data.get('attendance_access_token'),
                attendance_duration_since_supervisor_checkout=data.get('attendance_duration_since_supervisor_checkout')
            )
            
            attendance.save()
            
            return JsonResponse({
                'success': True,
                'attendance_id': attendance.id,
                'message': 'Attendance record created successfully'
            }, status=status.HTTP_201_CREATED)
        except Exception as e:
            return JsonResponse({
                'success': False,
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(["POST"])
def user_login(request):
    if request.method == "POST":
        try:
            data = request.POST

            username = data.get('username')
            password = data.get('password')

            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                curr_user = {
                    'username': user.username,
                    'first_name': user.first_name,
                    'last_name': user.last_name,
                    'email': user.email,
                    'role': 'supervisor'
                }
                return JsonResponse({'success': True, 'message': 'Login successful', 'user': curr_user}, status=status.HTTP_200_OK)
            else:
                return JsonResponse({'success': False, 'message': 'Invalid username or password'}, status=status.HTTP_401_UNAUTHORIZED)
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
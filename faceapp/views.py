from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.shortcuts import render, redirect
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from facenet_pytorch import InceptionResnetV1, MTCNN
from django.http import JsonResponse
from rest_framework import status
from PIL import Image
from .models import Worker
from .serializers import WorkerSerializer
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

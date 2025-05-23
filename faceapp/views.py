from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.shortcuts import render, redirect
from django.contrib import messages
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

def preprocess_image(img):
    """ Align and convert image to tensor format """
    img = img.convert("RGB")
    face = mtcnn(img)
    if face is None:
        return None
    return facenet(face.unsqueeze(0)).detach().numpy()[0]

def calculate_cosine_similarity(embedding1, embedding2):
    """ Compute cosine similarity between two embeddings """
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def post_face_to_compare(request):
    if request.method == "POST":
        img = request.FILES.get("image")
        if not img:
            return JsonResponse({"message": "Image is required.", "matchFound": False, "success": False}, status=status.HTTP_400_BAD_REQUEST)

        uploaded_embedding = preprocess_image(Image.open(img))
        if uploaded_embedding is None:
            return JsonResponse({"message": "No face detected.", "matchFound": False, "success": False}, status=status.HTTP_404_NOT_FOUND)

        workers = Worker.objects.all()
        if not workers:
            return JsonResponse({"message": "No registered workers available for comparison.", "matchFound": False, "success": False}, status=status.HTTP_404_NOT_FOUND)

        best_similarity = -1  
        best_worker = None
        threshold = 0.75 

        for worker in workers:
            try:
                worker_embeddings = np.array(worker.face_encoding) 
                if len(worker_embeddings.shape) == 1: 
                    worker_embeddings = worker_embeddings.reshape(1, -1)

                avg_worker_embedding = np.mean(worker_embeddings, axis=0)
                avg_worker_embedding /= np.linalg.norm(avg_worker_embedding)
                similarity = calculate_cosine_similarity(uploaded_embedding, avg_worker_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_worker = worker
            except Exception as e:
                print(f"Error processing worker {worker.name}: {e}")
                continue

        print("Best similarity:", round(best_similarity, 4))
        print("Best matched worker:", best_worker)

        if best_worker and best_similarity > threshold:
            return JsonResponse({"matched_worker": WorkerSerializer(best_worker).data, "similarity": round(best_similarity, 4), "matchFound": True, "success": True}, status=status.HTTP_200_OK)
        else:
            return JsonResponse({"message": "No match found.", "matchFound": False, "success": False}, status=status.HTTP_404_NOT_FOUND)

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from PIL import Image

from facenet_pytorch import InceptionResnetV1, MTCNN
from .models import Worker
from .serializers import WorkerSerializer

import numpy as np
import faiss

facenet = InceptionResnetV1(pretrained="vggface2").eval()
mtcnn = MTCNN()

index = None
worker_id_list = []

def build_faiss_index():
    """Builds the FAISS index with current worker embeddings."""
    global index, worker_id_list

    workers = Worker.objects.exclude(face_encoding=None)
    if not workers.exists():
        index = None
        worker_id_list = []
        return

    try:
        embeddings = [np.array(w.face_encoding, dtype='float32') for w in workers]
        if not embeddings:
            index = None
            worker_id_list = []
            return

        embeddings = np.vstack(embeddings)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        worker_id_list = [w.id for w in workers]
        print(f"✅ FAISS index built with {len(worker_id_list)} workers.")

    except Exception as e:
        print(f"❌ Error building FAISS index: {e}")
        index = None
        worker_id_list = []

build_faiss_index()

@api_view(["POST"])
def worker_registration(request):
    name = request.data.get("name")
    image = request.FILES.get("image")

    if not name or not image:
        return JsonResponse({"message": "Name and image are required.", "success": False}, status=status.HTTP_400_BAD_REQUEST)

    try:
        img = Image.open(image).convert("RGB")
        face = mtcnn(img)
        if face is None:
            return JsonResponse({"message": "No face detected.", "success": False}, status=status.HTTP_404_NOT_FOUND)

        embedding = facenet(face.unsqueeze(0)).detach().numpy()
        embedding /= np.linalg.norm(embedding)
        Worker.objects.create(name=name, face_encoding=embedding.tolist())

        build_faiss_index()

        return JsonResponse({"message": f"Worker '{name}' registered successfully.", "success": True}, status=status.HTTP_200_OK)
    except Exception as e:
        return JsonResponse({"message": str(e), "success": False}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(["POST"])
def post_face_to_compare(request):
    image = request.FILES.get("image")
    if not image:
        return JsonResponse({"message": "Image is required.", "matchFound": False, "success": False}, status=status.HTTP_400_BAD_REQUEST)

    if index is None or len(worker_id_list) == 0:
        return JsonResponse({"message": "No registered workers to compare.", "matchFound": False, "success": False}, status=status.HTTP_404_NOT_FOUND)

    try:
        img = Image.open(image).convert("RGB")
        face = mtcnn(img)
        if face is None:
            return JsonResponse({"message": "No face detected.", "matchFound": False, "success": False}, status=status.HTTP_404_NOT_FOUND)

        uploaded_embedding = facenet(face.unsqueeze(0)).detach().numpy().astype("float32")
        uploaded_embedding /= np.linalg.norm(uploaded_embedding)

        D, I = index.search(uploaded_embedding, 1)
        best_distance = float(D[0][0])
        best_idx = I[0][0]

        threshold = 0.75 
        if best_distance < threshold:
            matched_worker_id = worker_id_list[best_idx]
            matched_worker = Worker.objects.get(id=matched_worker_id)
            data = WorkerSerializer(matched_worker).data
            return JsonResponse({
                "matched_worker": data,
                "distance": round(best_distance, 4),
                "matchFound": True,
                "success": True
            }, status=status.HTTP_200_OK)

        else:
            return JsonResponse({"message": "No match found.", "matchFound": False, "success": False}, status=status.HTTP_404_NOT_FOUND)

    except Exception as e:
        return JsonResponse({"message": str(e), "matchFound": False, "success": False}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
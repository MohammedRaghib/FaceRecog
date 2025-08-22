from rest_framework.decorators import api_view
from rest_framework import status
from django.http import JsonResponse
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from .models import Worker
from .serializers import WorkerSerializer
import numpy as np
import faiss
import threading
import os

INDEX_FILE = "faces.index"

facenet = InceptionResnetV1(pretrained="vggface2").eval()
mtcnn = MTCNN(keep_all=True)

index = None
worker_id_list = []
index_lock = threading.Lock() 


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Row-wise normalization for cosine similarity."""
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def save_index():
    """Save FAISS index and worker IDs to disk."""
    with index_lock:
        if index is not None:
            faiss.write_index(index, INDEX_FILE)
            np.save("worker_ids.npy", np.array(worker_id_list))
            print(f"💾 FAISS index saved with {len(worker_id_list)} workers.")


def load_index():
    """Load FAISS index from disk, fallback to rebuild if missing."""
    global index, worker_id_list
    if os.path.exists(INDEX_FILE) and os.path.exists("worker_ids.npy"):
        index = faiss.read_index(INDEX_FILE)
        worker_id_list = np.load("worker_ids.npy").tolist()
        print(f"🔄 FAISS index loaded with {len(worker_id_list)} workers.")
        return True
    return False


def build_faiss_index():
    """Rebuild FAISS index from the database (used on first start or corruption)."""
    global index, worker_id_list

    workers = Worker.objects.exclude(face_encoding=None)
    if not workers.exists():
        index = None
        worker_id_list = []
        return

    embeddings = [np.array(w.face_encoding, dtype="float32") for w in workers]
    embeddings = np.vstack(embeddings)
    embeddings = normalize_embeddings(embeddings)

    with index_lock:
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        worker_id_list = [w.id for w in workers]

    save_index()


if not load_index():
    build_faiss_index()


@api_view(["POST"])
def worker_registration(request):
    name = request.data.get("name")
    image = request.FILES.get("image")

    if not name or not image:
        return JsonResponse({"message": "Name and image are required.", "success": False}, status=status.HTTP_400_BAD_REQUEST)

    try:
        img = Image.open(image).convert("RGB")
        faces, probs = mtcnn(img, return_prob=True)
        if faces is None or len(faces) == 0:
            return JsonResponse({"message": "No face detected.", "success": False}, status=status.HTTP_404_NOT_FOUND)

        best_idx = np.argmax(probs)
        face = faces[best_idx]

        embedding = facenet(face.unsqueeze(0)).detach().numpy().astype("float32")
        embedding = normalize_embeddings(embedding)

        worker = Worker.objects.create(name=name, face_encoding=embedding.tolist())

        with index_lock:
            if index is None:
                build_faiss_index()
            else:
                index.add(embedding)
                worker_id_list.append(worker.id)
                save_index()

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
        faces, probs = mtcnn(img, return_prob=True)
        if faces is None or len(faces) == 0:
            return JsonResponse({"message": "No face detected.", "matchFound": False, "success": False}, status=status.HTTP_404_NOT_FOUND)

        best_idx = np.argmax(probs)
        face = faces[best_idx]

        uploaded_embedding = facenet(face.unsqueeze(0)).detach().numpy().astype("float32")
        uploaded_embedding = normalize_embeddings(uploaded_embedding)

        with index_lock:
            similarity, I = index.search(uploaded_embedding, 1)

        best_similarity = float(similarity[0][0])
        best_idx = I[0][0]

        threshold = 0.7
        if best_similarity >= threshold:
            matched_worker_id = worker_id_list[best_idx]
            matched_worker = Worker.objects.get(id=matched_worker_id)
            data = WorkerSerializer(matched_worker).data
            return JsonResponse({
                "matched_worker": data,
                "similarity": round(best_similarity, 4),
                "matchFound": True,
                "success": True
            }, status=status.HTTP_200_OK)
        else:
            return JsonResponse({"message": "No match found.", "matchFound": False, "success": False}, status=status.HTTP_404_NOT_FOUND)

    except Exception as e:
        return JsonResponse({"message": str(e), "matchFound": False, "success": False}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

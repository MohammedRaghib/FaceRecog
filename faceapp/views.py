from django.http import JsonResponse
from .models import Person
import numpy as np
import hashlib
import threading
from functools import lru_cache
from typing import Tuple
from django.core.cache import cache
import cv2
import numpy as np
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from insightface.app import FaceAnalysis

MAX_SIDE = 1600
DEFAULT_THRESHOLD = 0.6
_CACHE_SIZE = 1024

_FACE_LOCK = threading.RLock()
_FACE_APP = None
_ONNX_PROVIDERS = [
    ["CUDAExecutionProvider", "CPUExecutionProvider"],
    ["CPUExecutionProvider"],
]

def _init_face_app() -> FaceAnalysis:
    global _FACE_APP
    if _FACE_APP is not None:
        return _FACE_APP

    last_err = None
    for providers in _ONNX_PROVIDERS:
        try:
            app = FaceAnalysis(name="buffalo_l", providers=providers)
            app.prepare(ctx_id=0, det_size=(640, 640))
            _FACE_APP = app
            return _FACE_APP
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to initialize FaceAnalysis: {last_err}")

def _read_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image")
    h, w = bgr.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def _sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()

@lru_cache(maxsize=_CACHE_SIZE)
def _embed_from_sha1(sha1_hex: str, raw_bytes: bytes) -> Tuple[bool, str, bytes]:
    app = _init_face_app()
    try:
        img = _read_image(raw_bytes)
    except Exception as e:
        return False, f"Decode error: {e}", b""

    with _FACE_LOCK:
        faces = app.get(img)
    if not faces:
        return False, "No face detected.", b""

    face = max(faces, key=lambda f: getattr(f, "det_score", 0.0))
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        return False, "Failed to compute embedding.", b""

    emb = emb.astype("float32")
    norm = np.linalg.norm(emb)
    if norm == 0.0:
        return False, "Zero-norm embedding.", b""
    emb /= norm

    return True, "ok", emb.tobytes()

def get_embedding(file_bytes: bytes) -> np.ndarray:
    h = _sha1(file_bytes)
    ok, msg, emb_bytes = _embed_from_sha1(h, file_bytes)
    if not ok:
        raise ValueError(msg)
    return np.frombuffer(emb_bytes, dtype=np.float32)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  

def update_person_face(person_id: int, emb: np.ndarray):
    person, created = Person.objects.get_or_create(national_id_number=person_id, defaults={
        "name": f"Person {person_id}",
        "national_id_number": f"ID{person_id}",
    })
    person.face_encoding = emb.tolist()
    person.save(update_fields=["face_encoding"])
    return person

def get_cached_faces():
    cache_key = "person_face_embeddings"
    cached = cache.get(cache_key)
    if cached:
        return cached
    qs = Person.objects.exclude(face_encoding=None)
    people = list(qs)
    if not people:
        return [], np.empty((0, 512), dtype=np.float32)
    embs = np.stack([np.array(p.face_encoding, dtype=np.float32) for p in people])
    cache.set(cache_key, (people, embs), timeout=30)
    return people, embs

class FaceVerifyView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        if "image1" not in request.FILES or "image2" not in request.FILES:
            return JsonResponse({"detail": "Both image1 and image2 are required."}, status=400)
        threshold = float(request.data.get("threshold", DEFAULT_THRESHOLD))
        try:
            emb1 = get_embedding(request.FILES["image1"].read())
            emb2 = get_embedding(request.FILES["image2"].read())
            sim = cosine_similarity(emb1, emb2)
            return JsonResponse({
                "similarity": sim,
                "threshold": threshold,
                "is_match": sim >= threshold
            })
        except ValueError as e:
            return JsonResponse({"detail": str(e)}, status=422)
        except Exception as e:
            return JsonResponse({"detail": f"Server error: {e}"}, status=500)

class FaceRegisterView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        if "image" not in request.FILES:
            return JsonResponse({"detail": "image is required."}, status=400)

        person_id = request.data.get("person_id", None)

        name = request.data.get("name", None)
        national_id_number = request.data.get("national_id_number", None)
        gender = request.data.get("gender", None)
        date_of_birth = request.data.get("date_of_birth", None)
        contact = request.data.get("contact", None)
        email = request.data.get("email", None)

        if not person_id and not (name and national_id_number):
            return JsonResponse({
                "detail": "Either person_id or (name and national_id_number) are required to create a person."
            }, status=400)

        try:
            emb = get_embedding(request.FILES["image"].read())

            if person_id:
                person, created = Person.objects.get_or_create(
                    pk=int(person_id),
                    defaults={
                        "name": name or f"Person {person_id}",
                        "national_id_number": national_id_number or f"ID{person_id}",
                        "gender": gender or "male",
                        "date_of_birth": date_of_birth or "2000-01-01",
                        "contact": contact,
                        "email": email,
                        "face_encoding": emb.tolist()
                    }
                )
                if not created:
                    person.face_encoding = emb.tolist()
                    person.save(update_fields=["face_encoding"])
            else:
                person = Person.objects.create(
                    name=name,
                    national_id_number=national_id_number,
                    gender=gender or "male",
                    date_of_birth=date_of_birth or "2000-01-01",
                    contact=contact,
                    email=email,
                    face_encoding=emb.tolist()
                )

            return JsonResponse({
                "person_id": person.id,
                "name": person.name,
                "detail": "Face encoding saved"
            })

        except ValueError as e:
            return JsonResponse({"detail": str(e)}, status=422)
        except Exception as e:
            return JsonResponse({"detail": f"Server error: {e}"}, status=500)
        
class FaceIdentifyView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        if "image" not in request.FILES:
            return JsonResponse({"detail": "image is required."}, status=400)
        topk = int(request.data.get("topk", 5))
        try:
            emb = get_embedding(request.FILES["image"].read())
            people, embs = get_cached_faces()
            if len(embs) == 0:
                return JsonResponse({"candidates": [], "detail": "No registered faces"})
            sims = np.dot(embs, emb)
            idx = np.argsort(-sims)[:topk]
            candidates = [
                {"person_id": people[i].id, "name": people[i].name, "similarity": float(sims[i])}
                for i in idx
            ]
            return JsonResponse({"candidates": candidates, "detail": "ok"})
        except ValueError as e:
            return JsonResponse({"detail": str(e)}, status=422)
        except Exception as e:
            return JsonResponse({"detail": f"Server error: {e}"}, status=500)
        
class FaceMultiIdentifyView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        if "image" not in request.FILES:
            return JsonResponse({"detail": "image is required."}, status=400)

        threshold = float(request.data.get("threshold", DEFAULT_THRESHOLD))
        topk = int(request.data.get("topk", 1)) 

        try:
            raw_bytes = request.FILES["image"].read()
            app = _init_face_app()
            img = _read_image(raw_bytes)

            with _FACE_LOCK:
                faces = app.get(img)

            if not faces:
                return JsonResponse({"faces": [], "detail": "No faces detected"})

            people, embs_db = get_cached_faces()
            if len(embs_db) == 0:
                return JsonResponse({"faces": [], "detail": "No registered faces"})

            results = []
            for face in faces:
                emb = getattr(face, "normed_embedding", None)
                if emb is None:
                    continue

                emb = emb.astype("float32")
                norm = np.linalg.norm(emb)
                if norm == 0.0:
                    continue
                emb /= norm

                sims = np.dot(embs_db, emb)
                idx = np.argsort(-sims)[:topk]

                candidates = [
                    {
                        "person_id": people[i].id,
                        "name": people[i].name,
                        "similarity": float(sims[i]),
                        "is_match": sims[i] >= threshold
                    }
                    for i in idx
                ]

                best_match = candidates[0] if candidates else None
                results.append({
                    "bbox": face.bbox.tolist(),
                    "best_match": best_match,
                    "candidates": candidates
                })

            return JsonResponse({"faces": results, "detail": "ok"})

        except ValueError as e:
            return JsonResponse({"detail": str(e)}, status=422)
        except Exception as e:
            return JsonResponse({"detail": f"Server error: {e}"}, status=500)
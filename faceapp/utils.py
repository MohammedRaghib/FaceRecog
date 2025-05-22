import faiss
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from .models import Worker

facenet = InceptionResnetV1(pretrained="vggface2").eval()
mtcnn = MTCNN()

embedding_dim = 512
index = faiss.IndexFlatL2(embedding_dim)

def get_worker_encoding(name, image_path):
    img = Image.open(image_path).convert("RGB")
    face = mtcnn(img)
    if face is None:
        print(f"Face not detected for {name}")
        return

    embedding = facenet(face.unsqueeze(0)).detach().numpy()

    Worker.objects.create(name=name, face_encoding=embedding.tolist())

    index.add(embedding)
    print(f"Worker {name} registered successfully!")
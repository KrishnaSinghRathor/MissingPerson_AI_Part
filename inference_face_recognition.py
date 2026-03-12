import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# ================= CONFIG =================
MODEL_PATH = "checkpoints2/face_recognition_model.pth"
IMG_SIZE = 112
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SIMILARITY_THRESHOLD = 0.5  # Cosine similarity threshold for recognition

# Face detection
PROTO = "deploy.prototxt"
CAFFEMODEL = "res10_300x300_ssd_iter_140000.caffemodel"
CONF_THRESHOLD = 0.6
# ==========================================


# ================= MODEL ARCHITECTURE =================
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.prelu = nn.PReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(x)
        out = self.prelu(out)
        return out


class ResNetFace(nn.Module):
    def __init__(self, num_layers=50, embedding_size=512):
        super(ResNetFace, self).__init__()
        
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34:
            layers = [3, 4, 6, 3]
        elif num_layers == 50:
            layers = [3, 4, 14, 3]
        else:
            raise ValueError(f"Invalid num_layers: {num_layers}")
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        
        self.layer1 = self._make_layer(64, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(256, 512, layers[3], stride=2)
        
        self.bn2 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(512 * 7 * 7, embedding_size)
        self.bn3 = nn.BatchNorm1d(embedding_size)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.bn2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn3(x)
        
        return x


# ================= FACE RECOGNITION CLASS =================
class FaceRecognizer:
    def __init__(self, model_path, device=DEVICE):
        self.device = device
        
        # Load model
        print("Loading face recognition model...")
        checkpoint = torch.load(model_path, map_location=device)
        embedding_size = checkpoint['embedding_size']
        
        self.model = ResNetFace(num_layers=50, embedding_size=embedding_size).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Face detector
        if os.path.exists(PROTO) and os.path.exists(CAFFEMODEL):
            self.face_net = cv2.dnn.readNetFromCaffe(PROTO, CAFFEMODEL)
        else:
            print("Warning: Face detector files not found. Face detection disabled.")
            self.face_net = None
        
        # Database for storing known faces
        self.face_database = {}  # {name: embedding}
        
        print("✅ Model loaded successfully!")
    
    # def detect_face(self, image):
    #     """Detect face in image and return cropped face"""
    #     if self.face_net is None:
    #         # If no detector, assume image is already cropped face
    #         return cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        
    #     h, w = image.shape[:2]
    #     blob = cv2.dnn.blobFromImage(
    #         cv2.resize(image, (300, 300)),
    #         1.0,
    #         (300, 300),
    #         (104.0, 177.0, 123.0)
    #     )
        
    #     self.face_net.setInput(blob)
    #     detections = self.face_net.forward()
        
    #     if detections.shape[2] == 0:
    #         return None
        
    #     confidence = detections[0, 0, 0, 2]
    #     if confidence < CONF_THRESHOLD:
    #         return None
        
    #     box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    #     x1, y1, x2, y2 = box.astype(int)
        
    #     # Add margin
    #     margin = 0.15
    #     bw = x2 - x1
    #     bh = y2 - y1
    #     mx = int(bw * margin)
    #     my = int(bh * margin)
        
    #     x1 = max(0, x1 - mx)
    #     y1 = max(0, y1 - my)
    #     x2 = min(w, x2 + mx)
    #     y2 = min(h, y2 + my)
        
    #     face = image[y1:y2, x1:x2]
    #     if face.size == 0:
    #         return None
        
    #     face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    #     return face


    def detect_face(self, image):
        """
        Detect multiple faces in image
        Returns: list of dicts with:
            {
                "face": cropped_face,
                "box": (x1, y1, x2, y2),
                "confidence": confidence
            }
        """

        faces = []

        if self.face_net is None:
            # Assume whole image is face
            resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            faces.append({
                "face": resized,
                "box": (0, 0, image.shape[1], image.shape[0]),
                "confidence": 1.0
            })
            return faces

        h, w = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        for i in range(detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence < CONF_THRESHOLD:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            # Add margin
            margin = 0.15
            bw = x2 - x1
            bh = y2 - y1
            mx = int(bw * margin)
            my = int(bh * margin)

            x1 = max(0, x1 - mx)
            y1 = max(0, y1 - my)
            x2 = min(w, x2 + mx)
            y2 = min(h, y2 + my)

            face = image[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

            faces.append({
                "face": face,
                "box": (x1, y1, x2, y2),
                "confidence": float(confidence)
            })

        return faces
    
    
    def get_embedding(self, face_image):
        """Get embedding vector for a face image"""
        # Convert BGR to RGB if needed
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(face_image)
        
        # Transform and add batch dimension
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Get embedding
        with torch.no_grad():
            embedding = self.model(img_tensor)
            embedding = nn.functional.normalize(embedding, dim=1)
        
        return embedding.cpu().numpy()[0]
    
    # def add_person(self, name, image_path):
    #     """Add a person to the database"""
    #     image = cv2.imread(image_path)
    #     if image is None:
    #         print(f"Error: Could not load image {image_path}")
    #         return False
        
    #     face = self.detect_face(image)
    #     if face is None:
    #         print(f"Error: No face detected in {image_path}")
    #         return False
        
    #     embedding = self.get_embedding(face)
    #     self.face_database[name] = embedding
    #     print(f"✅ Added {name} to database")
    #     return True



    def add_person(self, name, image_path):
        """Add a person to the database"""

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return False

        faces = self.detect_face(image)

        if not faces:
            print(f"Error: No face detected in {image_path}")
            return False

        # Take first detected face only
        face = faces[0]["face"]

        embedding = self.get_embedding(face)

        self.face_database[name] = embedding
        print(f"✅ Added {name} to database")
        return True
    
    def recognize_face(self, image_path):
        """Recognize a face in an image"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return None, 0.0
        
        face = self.detect_face(image)
        if face is None:
            print("Error: No face detected")
            return None, 0.0
        
        embedding = self.get_embedding(face)
        
        # Compare with database
        best_match = None
        best_similarity = -1.0
        
        for name, db_embedding in self.face_database.items():
            # Cosine similarity
            similarity = np.dot(embedding, db_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        if best_similarity >= SIMILARITY_THRESHOLD:
            return best_match, float(best_similarity)
        else:
            return "Unknown", float(best_similarity)
    
    def save_database(self, path="face_database.npz"):
        """Save face database to file"""
        np.savez(path, **self.face_database)
        print(f"✅ Database saved to {path}")
    
    def load_database(self, path="face_database.npz"):
        """Load face database from file"""
        if not os.path.exists(path):
            print(f"Error: Database file {path} not found")
            return False
        
        data = np.load(path)
        self.face_database = {name: data[name] for name in data.files}
        print(f"✅ Loaded {len(self.face_database)} persons from database")
        return True


# ================= EXAMPLE USAGE =================
def main():
    # Initialize recognizer
    recognizer = FaceRecognizer(MODEL_PATH)
    
    # Example 1: Add persons to database
    print("\n--- Adding persons to database ---")
    # recognizer.add_person("John Doe", "path/to/john.jpg")
    # recognizer.add_person("Jane Smith", "path/to/jane.jpg")
    
    # Save database
    # recognizer.save_database("face_database.npz")
    
    # Example 2: Load database and recognize
    print("\n--- Loading database and recognizing ---")
    # recognizer.load_database("face_database.npz")
    
    # Recognize a face
    # name, similarity = recognizer.recognize_face("path/to/test_image.jpg")
    # print(f"Recognized: {name} (similarity: {similarity:.4f})")
    
    print("\n✅ Inference script ready!")
    print("\nUsage:")
    print("1. Add persons to database using add_person()")
    print("2. Save database using save_database()")
    print("3. Recognize faces using recognize_face()")


if __name__ == "__main__":
    main()

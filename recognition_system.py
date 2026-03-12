import numpy as np
import faiss
import cv2
from inference_face_recognition import FaceRecognizer
from db_helper import Database
from datetime import datetime
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# points to: C:\Users\Krishna Singh\Desktop\missing_person
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# points to: C:\Users\Krishna Singh\Desktop\missing_person\backend
BACKEND_ROOT = os.path.join(PROJECT_ROOT, "backend")

class MissingPersonRecognitionSystem:
    """
    Core system that handles face recognition for missing persons
    
    This is the BRAIN of your system!
    """
   
    def __init__(self, model_path, db_config):
        """
        Initialize the recognition system
        
        Args:
            model_path: Path to trained face recognition model
            db_config: Dictionary with database connection details
        """
        print("\n" + "="*60)
        print("INITIALIZING MISSING PERSON RECOGNITION SYSTEM")
        print("="*60)
        
        # Step 1: Load face recognition model
        print("\n1️⃣ Loading face recognition model...")
        self.recognizer = FaceRecognizer(model_path)
        
        # Step 2: Connect to database
        print("\n2️⃣ Connecting to database...")
        self.db = Database(db_config)
        
        # Step 3: Initialize in-memory structures
        print("\n3️⃣ Initializing in-memory cache...")
        
        # FAISS index for fast similarity search
        # 512 = embedding dimension
        self.embedding_dim = 512
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Cache to store person details (instant lookup!)
        # Structure: {case_id: {name, phone, location, etc.}}
        self.metadata_cache = {}
        
        # Mapping from FAISS index position to case_id
        # Example: index 0 → "CASE_001", index 1 → "CASE_002"
        self.index_to_case_id = []
        
        # Step 4: Load all approved cases from database into memory
        print("\n4️⃣ Loading approved cases into memory...")
        self.load_all_cases()
        
        print("\n" + "="*60)
        print("✅ SYSTEM READY!")
        print(f"   Monitoring for {len(self.metadata_cache)} missing persons")
        print("="*60 + "\n")
    
    def load_all_cases(self):
        """
        Load all approved cases from database into memory
        
        This runs at startup and loads everything into RAM for speed!
        """
        # Get all approved cases from database
        cases = self.db.get_approved_cases()
        
        if not cases:
            print("   No approved cases found in database")
            return
        
        print(f"   Found {len(cases)} approved cases")
        
        embeddings_list = []
        
        for case in cases:
            case_id = case['case_id']
            embedding = case['embedding']
            
            if embedding is None:
                print(f"   ⚠️ Skipping {case_id} (no embedding)")
                continue
            
            # Add embedding to list (will add to FAISS later)
            embeddings_list.append(embedding)
            
            # Store metadata in cache
            self.metadata_cache[case_id] = {
                'name': case['name'],
                'phone_number': case['phone_number'],
                'location': case['last_seen_location'],
                'photo_paths': case['photo_paths']
            }
            
            # Map FAISS index position to case_id
            self.index_to_case_id.append(case_id)
        
        # Build FAISS index from all embeddings at once (fast!)
        if embeddings_list:
            embeddings_array = np.array(embeddings_list).astype('float32')
            self.faiss_index.add(embeddings_array)
            
            print(f"   ✅ Built FAISS index with {self.faiss_index.ntotal} embeddings")
            print(f"   ✅ Loaded {len(self.metadata_cache)} cases into cache")
    
    def generate_embedding_from_photos(self, photo_paths):
        """
        Generate a robust embedding from multiple photos
        
        Why multiple photos?
        - More robust to different angles
        - More robust to different lighting
        - Average = better generalization
        
        Args:
            photo_paths: List of 2-4 photo file paths
            
        Returns:
            embedding: 512-dimensional numpy array (or None if failed)
        """
        print(f"\n   Processing {len(photo_paths)} photos...")

        embeddings = []

        for i, photo_path in enumerate(photo_paths, 1):
            print(f"   Photo {i}/{len(photo_paths)}: {photo_path}")

            normalized_path = photo_path.replace("\\", "/")

            full_path = (
                normalized_path
                if os.path.isabs(normalized_path)
                else os.path.join(BACKEND_ROOT, normalized_path)
            )

            print(f"      📂 Trying path: {full_path}")
            print(f"      📂 Exists: {os.path.exists(full_path)}")

            image = cv2.imread(full_path)
            if image is None:
                print("      ❌ Could not read image")
                continue

            # ✅ FIXED SECTION
            faces = self.recognizer.detect_face(image)

            if not faces:
                print(f"      ⚠️ No face detected")
                continue

            face = faces[0]["face"]

            embedding = self.recognizer.get_embedding(face)
            embeddings.append(embedding)

            print(f"      ✅ Embedding generated")

        if not embeddings:
            print(f"   ❌ No valid embeddings generated")
            return None

        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        print(f"   ✅ Final embedding: average of {len(embeddings)} photos")

        return avg_embedding
    
    def add_missing_person(self, case_id, photo_paths):
        """
        Add a new missing person to the system
        
        This is called when admin approves a case!
        
        Args:
            case_id: Unique case identifier from database
            photo_paths: List of 2-4 uploaded photos
            
        Returns:
            success: True if added successfully, False otherwise
        """
        print(f"\n{'='*60}")
        print(f"ADDING NEW MISSING PERSON: {case_id}")
        print(f"{'='*60}")
        
        # Step 1: Generate embedding from photos
        print("\n📸 Generating face embedding...")
        embedding = self.generate_embedding_from_photos(photo_paths)
        
        if embedding is None:
            print(f"\n❌ Failed to generate embedding for {case_id}")
            return False
        
        # Step 2: Save embedding to database
        print("\n💾 Saving to database...")
        self.db.save_embedding(case_id, embedding)
        
        # Step 3: Get full case details
        print("\n📋 Fetching case details...")
        case_data = self.db.get_case_details(case_id)
        
        if not case_data:
            print(f"❌ Could not fetch case details for {case_id}")
            return False
        
        # Step 4: Add to FAISS index (in-memory, fast search!)
        print("\n🔍 Adding to search index...")
        embedding_reshaped = embedding.reshape(1, -1).astype('float32')
        self.faiss_index.add(embedding_reshaped)
        
        # Step 5: Add to metadata cache (in-memory, instant lookup!)
        print("\n💾 Adding to memory cache...")
        self.metadata_cache[case_id] = {
            'name': case_data['name'],
            'phone_number': case_data['phone_number'],
            'location': case_data['last_seen_location'],
            'photo_paths': case_data['photo_paths']
        }
        
        # Step 6: Update index mapping
        self.index_to_case_id.append(case_id)
        
        print(f"\n{'='*60}")
        print(f"✅ SUCCESS!")
        print(f"   Case {case_id} ({case_data['name']}) added to system")
        print(f"   Total cases in system: {len(self.metadata_cache)}")
        print(f"{'='*60}\n")
        
        return True
    
    def search_face(self, embedding, threshold=0.4):
        """
        Search for a face in the missing persons database
        
        This is called for every detected face in video!
        
        Args:
            embedding: 512-dim face embedding
            threshold: Minimum similarity to consider a match (0.0-1.0)
                      0.4 = 40% similar (lenient for missing persons)
                      0.7 = 70% similar (strict)
            
        Returns:
            found: True if match found, False otherwise
            case_data: Dictionary with person details (or None)
            confidence: Similarity score (0.0-1.0)
        """
        # Check if we have any cases in the system
        if self.faiss_index.ntotal == 0:
            return False, None, 0.0
        
        # Reshape embedding for FAISS
        embedding_reshaped = embedding.reshape(1, -1).astype('float32')
        
        # Search in FAISS index
        # k=1 means find the 1 closest match
        distances, indices = self.faiss_index.search(embedding_reshaped, k=1)
        
        # Get distance and index
        distance = distances[0][0]  # L2 distance
        index_pos = indices[0][0]    # Position in FAISS index
        
        # Convert L2 distance to similarity score
        # Lower distance = Higher similarity
        # Formula: similarity = 1 / (1 + distance)
        # Distance 0 → Similarity 1.0 (perfect match)
        # Distance 10 → Similarity 0.09 (very different)
        similarity = 1.0 / (1.0 + distance)
        
        # Check if similarity is above threshold
        if similarity >= threshold and index_pos < len(self.index_to_case_id):
            # We found a match!
            case_id = self.index_to_case_id[index_pos]
            
            # Get details from cache (instant!)
            case_data = self.metadata_cache[case_id].copy()
            case_data['case_id'] = case_id
            
            return True, case_data, similarity
        
        # No match found
        return False, None, similarity
    
    # def process_frame(self, frame, location="Unknown", camera_id="CAM_001"):
    #     """
    #     Process a single video frame
        
    #     Args:
    #         frame: OpenCV image/frame
    #         location: Where this frame is from
    #         camera_id: Camera identifier
            
    #     Returns:
    #         matches: List of detected persons (usually 0 or 1)
    #     """
    #     # Step 1: Detect face in frame
    #     face = self.recognizer.detect_face(frame)
        
    #     if face is None:
    #         # No face detected in this frame
    #         return []
        
    #     # Step 2: Generate embedding for detected face
    #     embedding = self.recognizer.get_embedding(face)
        
    #     # Step 3: Search in database
    #     found, case_data, confidence = self.search_face(embedding)
        
    #     if found:
    #         # Step 4: We found a missing person!
    #         print(f"\n🚨 MATCH FOUND!")
    #         print(f"   Person: {case_data['name']}")
    #         print(f"   Confidence: {confidence*100:.2f}%")
    #         print(f"   Location: {location}")
            
    #         # Step 5: Save detection frame
    #         frame_path = self.save_detection_frame(
    #             frame, case_data, confidence, location
    #         )
            
    #         # Step 6: Record in database
    #         detection_id = self.db.record_detection(
    #             case_id=case_data['case_id'],
    #             location=location,
    #             camera_id=camera_id,
    #             confidence=confidence,
    #             frame_path=frame_path
    #         )
            
    #         # Return match information
    #         return [{
    #             'detection_id': detection_id,
    #             'case_id': case_data['case_id'],
    #             'name': case_data['name'],
    #             'phone_number': case_data['phone_number'],
    #             'confidence': confidence,
    #             'location': location,
    #             'camera_id': camera_id,
    #             'frame_path': frame_path
    #         }]
        
    #     return []
    

    def process_frame(self, frame, location="Unknown", camera_id="CAM_001"):

        matches = []

        faces = self.recognizer.detect_face(frame)

        if not faces:
            return []

        frame_saved = False
        frame_path = None

        for face_data in faces:

            face = face_data["face"]
            x1, y1, x2, y2 = face_data["box"]

            # Generate embedding
            embedding = self.recognizer.get_embedding(face)

            # Search in DB
            found, case_data, confidence = self.search_face(embedding)

            print("Confidence:", confidence)

            if found:

                print(f"\n🚨 MATCH FOUND!")
                print(f"   Person: {case_data['name']}")
                print(f"   Confidence: {confidence*100:.2f}%")
                print(f"   Location: {location}")

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    f"{case_data['name']} ({confidence*100:.1f}%)",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

                # Save frame ONLY ONCE
                if not frame_saved:
                    frame_path = self.save_detection_frame(
                        frame, case_data, confidence, location
                    )
                    frame_saved = True

                # Record detection
                detection_id = self.db.record_detection(
                    case_id=case_data['case_id'],
                    location=location,
                    camera_id=camera_id,
                    confidence=confidence,
                    frame_path=frame_path
                )

                matches.append({
                    'detection_id': detection_id,
                    'case_id': case_data['case_id'],
                    'name': case_data['name'],
                    'phone_number': case_data['phone_number'],
                    'confidence': confidence,
                    'location': location,
                    'camera_id': camera_id,
                    'frame_path': frame_path
                })

        return matches
    def save_detection_frame(self, frame, case_data, confidence, location):
        """
        Save frame where person was detected
        
        Args:
            frame: OpenCV image
            case_data: Person details
            confidence: Match confidence
            location: Where detected
            
        Returns:
            frame_path: Path to saved image
        """
        # Create detections folder if it doesn't exist
        os.makedirs("detections", exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_{case_data['case_id']}_{timestamp}.jpg"
        frame_path = os.path.join("detections", filename)
        
        # Add text overlay to frame
        text = f"{case_data['name']} - {confidence*100:.1f}%"
        cv2.putText(frame, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        text2 = f"Location: {location}"
        cv2.putText(frame, text2, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save frame
        cv2.imwrite(frame_path, frame)
        
        print(f"   💾 Detection frame saved: {frame_path}")
        
        return frame_path


# Example usage
if __name__ == "__main__":
    # Database configuration
    db_config = {
        'host': 'localhost',
        'user': 'your_user',
        'password': 'your_password',
        'database': 'missing_persons_db'
    }
    
    # Initialize system
    system = MissingPersonRecognitionSystem(
        model_path="checkpoints2/face_recognition_model.pth",
        db_config=db_config
    )
    
    # Test: Add a new missing person (simulate admin approval)
    system.add_missing_person(
        case_id="CASE_001",
        photo_paths=[
            "uploads/case_001/photo1.jpg",
            "uploads/case_001/photo2.jpg"
        ]
    )
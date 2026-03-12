from flask import Flask, request, jsonify
from flask_cors import CORS
from recognition_system import MissingPersonRecognitionSystem
import os

app = Flask(__name__)
CORS(app)  # Allow requests from Node.js

# ============================================================
# INITIALIZE RECOGNITION SYSTEM
# ============================================================

# PostgreSQL configuration
db_config = {
    'host': 'localhost',
    'database': 'missing_person_db',
    'user': 'postgres',  # Change to your username
    'password': '@10kechar',  # Change to your password
    'port': 5432
}

# Initialize the recognition system (loads all approved cases)
print("🚀 Starting Flask API...")
recognition_system = MissingPersonRecognitionSystem(
    model_path="checkpoints/face_recognition_model.pth",
    db_config=db_config
)
print("✅ Flask API ready!")

# ============================================================
# API ENDPOINTS
# ============================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Check if API is running"""
    return jsonify({
        'status': 'healthy',
        'total_cases': len(recognition_system.metadata_cache),
        'message': 'Face recognition API is running'
    })

@app.route('/api/generate-embedding', methods=['POST'])
def generate_embedding():
    """
    Generate embedding when admin approves a case
    
    This is called by Node.js backend!
    
    Expected JSON:
    {
        "case_id": "CASE_123",
        "photo_paths": ["uploads/case_123/photo1.jpg", "uploads/case_123/photo2.jpg"]
    }
    """
    try:
        data = request.json
        
        case_id = data.get('case_id')
        photo_paths = data.get('photo_paths')
        
        if not case_id or not photo_paths:
            return jsonify({
                'success': False,
                'error': 'case_id and photo_paths are required'
            }), 400
        
        if len(photo_paths) < 2:
            return jsonify({
                'success': False,
                'error': 'At least 2 photos are required'
            }), 400
        
        print(f"\n📸 Generating embedding for {case_id}")
        print(f"   Photos: {photo_paths}")
        
        # Generate embedding and add to system
        success = recognition_system.add_missing_person(
            case_id=case_id,
            photo_paths=photo_paths
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Embedding generated and saved for {case_id}',
                'total_cases': len(recognition_system.metadata_cache)
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to generate embedding. Check if photos contain faces.'
            }), 400
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/reload-cases', methods=['POST'])
def reload_cases():
    """Reload all approved cases into memory"""
    try:
        print("\n🔄 Reloading all cases...")
        recognition_system.load_all_cases()
        
        return jsonify({
            'success': True,
            'message': 'Cases reloaded successfully',
            'total_cases': len(recognition_system.metadata_cache)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/search-face', methods=['POST'])
def search_face_endpoint():
    """
    Search for a face (for testing)
    
    Expected: multipart/form-data with 'image' file
    """
    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        file = request.files['image']
        
        # Save temporarily
        import cv2
        import numpy as np
        from werkzeug.utils import secure_filename
        
        filename = secure_filename(file.filename)
        temp_path = f"temp_{filename}"
        file.save(temp_path)
        
        # Read image
        image = cv2.imread(temp_path)
        
        # Detect face
        face = recognition_system.recognizer.detect_face(image)
        
        if face is None:
            os.remove(temp_path)
            return jsonify({
                'success': False,
                'error': 'No face detected in image'
            }), 400
        
        # Generate embedding
        embedding = recognition_system.recognizer.get_embedding(face)
        
        # Search
        found, case_data, confidence = recognition_system.search_face(embedding)
        
        # Clean up
        os.remove(temp_path)
        
        if found:
            return jsonify({
                'success': True,
                'found': True,
                'case_id': case_data['case_id'],
                'name': case_data['name'],
                'confidence': float(confidence),
                'location': case_data['location']
            })
        else:
            return jsonify({
                'success': True,
                'found': False,
                'message': 'No match found in database'
            })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("FACE RECOGNITION FLASK API")
    print("="*60)
    print("Endpoints:")
    print("  GET  /health                    - Health check")
    print("  POST /api/generate-embedding    - Generate embedding (called by Node.js)")
    print("  POST /api/reload-cases          - Reload all cases")
    print("  POST /api/search-face           - Test face search")
    print("="*60 + "\n")
    
    # Run on port 5001 (Node.js uses 5000)
    app.run(host='0.0.0.0', port=5001, debug=True)
import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
import face_recognition
from datetime import datetime

app = Flask(__name__)

# Load known encodings and student IDs
path = 'AttendanceImages'
encodeListKnown = []  # Load known encodings here
studentIds = []  # Load student IDs here

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def load_known_encodings():
    global encodeListKnown, studentIds
    images = []
    myList = os.listdir(path)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        studentIds.append(os.path.splitext(cl)[0])
    encodeListKnown = findEncodings(images)
    #print('Encoding Complete')

def recognize_student(image):
    # Find face encodings in the uploaded image
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(imgRGB)
    
    if len(face_encodings) == 0:
        return None
    
    # Compare with known encodings
    for encodeFace in face_encodings:
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        face_distances = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        # Find the best match
        if True in matches:
            match_index = matches.index(True)
            student_id = studentIds[match_index].upper()
            
            # Get current time        
            now = datetime.now()
            current_time = now.strftime('%H:%M:%S')
            
            return student_id, current_time
    
    return None

@app.route('/recognize', methods=['POST'])
def recognize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Read image file
    img_np = np.fromstring(file.read(), np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    
    result = recognize_student(img)
    if result:
        student_id, current_time = result
        return jsonify({'StudentCode': student_id, 'Time': current_time})
    else:
        return jsonify({'error': 'Student not recognized'})

if __name__ == '__main__':
    load_known_encodings()
  
    app.run(debug=True)

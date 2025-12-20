import cv2
import face_recognition
import pickle
import os

# 1. Setup Folder Path
folderPath = 'Images'
if not os.path.exists(folderPath):
    print(f"Error: The folder '{folderPath}' does not exist. Please create it.")
    exit()

pathList = os.listdir(folderPath)
imgList = []
studentIDs = []

print(f"Files found in folder: {pathList}")

# 2. Importing student images with strict filtering
for path in pathList:
    # SKIP hidden files like .DS_Store or non-image files
    if not path.startswith('.') and path.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.imread(os.path.join(folderPath, path))
        if img is not None:
            imgList.append(img)
            # Take only the name, remove the extension (e.g., '12345.png' -> '12345')
            studentIDs.append(os.path.splitext(path)[0])
        else:
            print(f"Warning: Could not load image {path}")

print(f"Validated Student IDs to encode: {studentIDs}")


# 3. Encoding Function with Mismatch Prevention
def findEncodings(imagesList, idsList):
    encodeList = []
    finalIds = []  # New list to ensure IDs match ONLY successfully encoded faces

    for img, student_id in zip(imagesList, idsList):
        # Convert BGR (OpenCV) to RGB (Face Recognition)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect face and encode
        encodings = face_recognition.face_encodings(img)

        if len(encodings) > 0:
            encodeList.append(encodings[0])
            finalIds.append(student_id)
            print(f"Encoded: {student_id}")
        else:
            print(f"Error: No face found in image for {student_id}. This ID will be skipped.")

    return encodeList, finalIds


print("Encoding process started...")
# Pass both lists to keep them synchronized
encodeListKnown, studentIDsFinal = findEncodings(imgList, studentIDs)

# Combine the face vectors and the IDs into one list
encodeListKnownWithIds = [encodeListKnown, studentIDsFinal]
print(f"Encoding complete. Successfully mapped {len(studentIDsFinal)} faces.")

# 4. Save to Pickle File
file = open("encodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()

print("----------------------------")
print("File Saved: encodeFile.p")
print("You can now run main.py")
print("----------------------------")
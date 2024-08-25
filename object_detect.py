# import cv2
# import numpy as np

# # Load YOLO
# net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# # Load COCO class labels
# with open("coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     height, width, channels = frame.shape

#     # Prepare the image for YOLO
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(output_layers)

#     # Process detections
#     class_ids = []
#     confidences = []
#     boxes = []

#     for out in outs:
#         for detection in out:
#             for obj in detection:
#                 obj = np.array(obj).flatten()  # Ensure it's a flat array

#                 if len(obj) < 6:
#                     continue  # Skip if object array is too short

#                 # Extract class scores and objectness
#                 scores = obj[5:]
#                 if len(scores) == 0:
#                     continue  # Skip if no scores are present

#                 class_id = int(np.argmax(scores))
#                 confidence = float(obj[4])

#                 if confidence > 0.5:
#                     # Convert center coordinates to bounding box coordinates
#                     center_x, center_y, w, h = obj[:4]
#                     center_x *= width
#                     center_y *= height
#                     w *= width
#                     h *= height
#                     x = int(center_x - w / 2)
#                     y = int(center_y - h / 2)

#                     boxes.append([x, y, int(w), int(h)])
#                     class_ids.append(class_id)
#                     confidences.append(confidence)

#     # Apply non-max suppression to reduce overlapping boxes
#     indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#     if len(indices) > 0:
#         indices = indices.flatten()

#     for i in indices:
#         box = boxes[i]
#         x, y, w, h = box
#         label = str(classes[class_ids[i]])
#         confidence = confidences[i]
#         color = (0, 255, 0)  # Green color for bounding box
#         cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#         cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     cv2.imshow("Image", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    
    # Break the loop if frame is not retrieved
    if not ret:
        print("Failed to grab frame")
        break

    height, width, channels = frame.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            for obj in detection:
                obj = np.array(obj)
                
                if obj.size == 0:
                    continue
                
                # Ensure obj is a 1D array and has enough elements
                if obj.ndim == 1 and len(obj) >= 6:
                    obj = obj.flatten()  # Flatten to ensure it's a flat array
                    
                    # Extract class scores and objectness
                    scores = obj[5:]
                    if len(scores) == 0:
                        continue  # Skip if no scores are present

                    class_id = int(np.argmax(scores))
                    confidence = obj[4]

                    if confidence > 0.5:
                        # Convert center coordinates to bounding box coordinates
                        center_x, center_y, w, h = obj[:4]
                        center_x *= width
                        center_y *= height
                        w *= width
                        h *= height
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, int(w), int(h)])
                        class_ids.append(class_id)
                        confidences.append(float(confidence))

    # Apply non-max suppression to reduce overlapping boxes
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indices) > 0:
            indices = indices.flatten()

        for i in indices:
            box = boxes[i]
            x, y, w, h = box
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Print class and confidence to the console
            print(f"Detected: {label} with confidence: {confidence:.2f}")

    # Display the result
    cv2.imshow("Image", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
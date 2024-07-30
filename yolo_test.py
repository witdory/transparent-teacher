import numpy as np
import cv2

# Load the class names
with open('./content/coco.names.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Generate colors for the classes
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Read the image
img = cv2.imread('./content/teacher.png')
height, width, channels = img.shape

# Prepare the image for YOLO
blob = cv2.dnn.blobFromImage(img, 1.0 / 256, (448, 448), (0, 0, 0), swapRB=True, crop=False)

# Load YOLO model
yolo_model = cv2.dnn.readNet('./yolov3.weights', './content/yolov3.cfg')

# Get layer names
layer_names = yolo_model.getLayerNames()

# Ensure getUnconnectedOutLayers returns the correct indices
try:
    out_layers = [layer_names[i - 1] for i in yolo_model.getUnconnectedOutLayers().flatten()]
except AttributeError:  # Handle the case where getUnconnectedOutLayers() returns a list of lists
    out_layers = [layer_names[i[0] - 1] for i in yolo_model.getUnconnectedOutLayers()]

# Perform forward pass
yolo_model.setInput(blob)
output3 = yolo_model.forward(out_layers)

# Initialize lists for detected objects
class_ids, confidences, boxes = [], [], []

# Process the detections
for output in output3:
    for vec85 in output:
        scores = vec85[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            centerx, centery = int(vec85[0] * width), int(vec85[1] * height)
            w, h = int(vec85[2] * width), int(vec85[3] * height)
            x, y = int(centerx - w / 2), int(centery - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-maximum suppression to remove overlapping boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw the boxes on the image
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        text = f"{classes[class_ids[i]]}: {confidences[i]:.3f}"
        cv2.rectangle(img, (x, y), (x + w, y + h), colors[class_ids[i]], 2)
        cv2.putText(img, text, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, colors[class_ids[i]], 2)

# Display the image
cv2.imshow('Object detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

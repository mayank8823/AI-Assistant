import os
from inference_sdk import InferenceHTTPClient
import cv2
import pyttsx3

# Initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="ZQRfTgadHjzyvprPkvd3"
)

def process_and_announce(image_path):
    # Read the image
    image = cv2.imread(image_path)
    result = CLIENT.infer(image_path, model_id="threat-w1us9/1")

    if image is None:
        print("Error: Image not found.")
        return

    # Get the dimensions of the image
    height, width, _ = image.shape

    # Initialize a list to hold detection results for speech
    detections = []

    # Loop through predictions and draw the bounding box
    for prediction in result['predictions']:
        x_center = prediction['x']
        y_center = prediction['y']
        bbox_width = prediction['width']
        bbox_height = prediction['height']
        
        # Calculate the top-left and bottom-right coordinates of the bounding box
        x1 = int(x_center - bbox_width / 2)
        y1 = int(y_center - bbox_height / 2)
        x2 = int(x_center + bbox_width / 2)
        y2 = int(y_center + bbox_width / 2)
        
        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Optionally, you can also put the class name and confidence score
        label = f"{prediction['class']} ({prediction['confidence']:.2f})"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add the detection to the list
        detections.append(f"{prediction['class']} with confidence {prediction['confidence']:.2f}")

    # Display the image
    if detections:
        message = "This image contains " + ", ".join(detections)
    else:
        message = "No detections in this image."
    engine = pyttsx3.init()
    engine.say(message)
    
    cv2.imshow('Detection', image)
    
    cv2.waitKey(0)
    engine.runAndWait()
    cv2.destroyAllWindows()
    
    # Convert the detections to speech
    

    # Initialize the pyttsx3 engine
    

# Folder containing images
image_folder = 'chec'

# Process each image in the folder
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_folder, filename)
        process_and_announce(image_path)

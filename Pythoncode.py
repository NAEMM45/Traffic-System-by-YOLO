import cv2
import os
import torch

# YOLOv5 Model Loading
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load the YOLOv5 model

# Define the path to the image (replace input with direct path)
image_path = "C:/Users/naeem/VS Code Programs/Python_Projects/SIH/images.jpeg"

# Function to process the image for traffic signal control
def process_traffic_image(image_path):
    print("Checking if the image path exists...")

    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: The image path '{image_path}' does not exist.")
        return

    print("Loading image...")
    image = cv2.imread(image_path)
    
    # Error handling if image loading fails
    if image is None:
        print("Error: Unable to load image. Please check the file path.")
        return

    print("Image loaded successfully!")
    
    # YOLOv5 inference
    print("Performing YOLOv5 inference on the image...")
    results = model(image)
    
    # Get the pandas dataframe of detected objects
    detected_objects = results.pandas().xyxy[0]
    
    # Filter the DataFrame to count vehicles (this includes common vehicle types)
    vehicle_classes = ['car', 'bus', 'truck', 'motorcycle']  # Add any other vehicle types you want
    vehicle_count = detected_objects[detected_objects['name'].isin(vehicle_classes)].shape[0]

    print(f"Number of vehicles detected: {vehicle_count}")
    
    # Displaying results
    print("Inference complete. Rendering results...")
    results.show()  # Display YOLOv5 detection results
    
    # Proceed with lane detection, vehicle detection, etc.
    lane_image = detect_lanes(image)
    print("Lane detection complete")
    
    traffic_density = detect_traffic_density(image)
    print(f"Traffic density: {traffic_density}")
    
    emergency_detected = detect_emergency_vehicle(image)
    print(f"Emergency detected: {emergency_detected}")
    
    vehicle_image = detect_vehicles(lane_image)
    print("Vehicle detection complete")
    
    # Control traffic signal based on detected parameters
    signal = control_traffic_signal(traffic_density, emergency_detected)
    print(f"Traffic Signal: {signal}")
    
    # Show the processed image with detections and vehicle count
    text = f"Vehicles Detected: {vehicle_count}"
    font_scale = 1
    font_thickness = 2
    text_color = (0, 255, 0)

    # Get the size of the image
    image_height, image_width = vehicle_image.shape[:2]

    # Calculate the size of the text box
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

    # Set the text position near the top left but within bounds
    x_position = min(50, image_width - text_width - 10)  # Adjust to make sure text stays within the frame
    y_position = min(50, image_height - text_height - 10)

    # Draw the vehicle count text within the image boundaries
    cv2.putText(vehicle_image, text, (x_position, y_position), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    print("Displaying processed image...")
    cv2.imshow("Traffic Control with Lane and Vehicle Detection", vehicle_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to draw bounding boxes within the frame
def draw_bbox_within_frame(image, bbox, label, confidence, font_scale=0.5, thickness=2):
    # Get the image dimensions
    img_h, img_w = image.shape[:2]
    
    # Unpack bbox values
    x1, y1, x2, y2 = map(int, bbox[:4])

    # Adjust label position to stay within bounds
    text = f"{label} {confidence:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    # Ensure the label stays within the frame
    x_label = max(0, min(x1, img_w - label_width))
    y_label = max(label_height, y1 - 5)
    
    # Draw the bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness)
    
    # Draw the label
    cv2.putText(image, text, (x_label, y_label), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

# Dummy functions for lane detection, traffic density, and emergency vehicle detection
def detect_lanes(image):
    # Dummy lane detection code (replace with actual implementation)
    print("Detecting lanes...")
    return image

def detect_traffic_density(image):
    # Dummy traffic density calculation (replace with actual implementation)
    print("Calculating traffic density...")
    return "Low"

def detect_emergency_vehicle(image):
    # Dummy emergency vehicle detection (replace with actual implementation)
    print("Detecting emergency vehicles...")
    return False

def detect_vehicles(image):
    # Dummy vehicle detection (replace with actual implementation)
    print("Detecting vehicles...")
    return image

def control_traffic_signal(traffic_density, emergency_detected):
    # Dummy traffic signal control logic (replace with actual implementation)
    print("Controlling traffic signal...")
    if emergency_detected:
        return "Green for emergency vehicle"
    elif traffic_density == "High":
        return "Green for longer duration"
    else:
        return "Standard signal"

# Process the image
process_traffic_image(image_path)

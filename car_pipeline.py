from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import os

# Updated damage types and cost ranges to match your model's classes
DAMAGE_COST_MAP = {
    0: ('crack_and_hole', 100, 300),
    1: ('medium_deformation', 200, 500),
    2: ('severe_deformation', 500, 1000),
    3: ('severe_scratch', 100, 300),
    4: ('slight_deformation', 50, 150),
    5: ('slight_scratch', 20, 80),
    6: ('windshield_damage', 200, 400)
}

def load_model():
    try:
        model_path = r"C:\Users\yashs\Downloads\Caryolo\Caryolo\best (4).pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        model = YOLO(model_path)
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def load_image(image_path):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        img = Image.open(image_path)
        img = img.convert('RGB')
        return np.array(img)
    except Exception as e:
        raise Exception(f"Error loading image: {str(e)}")

def detect_damages(image, model):
    try:
        results = model(image)
        damage_info = []

        for detection in results[0].boxes:
            class_id = int(detection.cls.cpu().numpy())
            confidence = float(detection.conf.cpu().numpy())
            bbox = detection.xyxy.cpu().numpy()

            # Check if class_id exists in DAMAGE_COST_MAP
            if class_id in DAMAGE_COST_MAP:
                damage_info.append({
                    "class_id": class_id,
                    "damage_type": DAMAGE_COST_MAP[class_id][0],
                    "confidence": confidence,
                    "bbox": bbox.flatten()
                })
            else:
                print(f"Warning: Unknown class_id {class_id} detected")

        return results, damage_info
    except Exception as e:
        raise Exception(f"Error detecting damages: {str(e)}")

def estimate_cost(damage_results):
    try:
        total_min_cost = 0
        total_max_cost = 0
        cost_breakdown = []

        for damage in damage_results:
            damage_type = damage['damage_type']
            confidence = damage['confidence']
            
            # Find class_id from damage_type
            class_id = None
            for id, (type_name, _, _) in DAMAGE_COST_MAP.items():
                if type_name == damage_type:
                    class_id = id
                    break
            
            if class_id is None:
                print(f"Warning: Unknown damage type {damage_type}")
                continue

            min_cost = DAMAGE_COST_MAP[class_id][1]
            max_cost = DAMAGE_COST_MAP[class_id][2]

            # Calculate estimated cost range based on confidence
            estimated_min = min_cost * confidence
            estimated_max = max_cost * confidence

            total_min_cost += estimated_min
            total_max_cost += estimated_max

            cost_breakdown.append({
                "type": damage_type,
                "confidence": confidence,
                "min_cost": round(estimated_min, 2),
                "max_cost": round(estimated_max, 2)
            })

        return {
            "min_cost": round(total_min_cost, 2),
            "max_cost": round(total_max_cost, 2),
            "currency": "USD",
            "breakdown": cost_breakdown
        }
    except Exception as e:
        raise Exception(f"Error estimating cost: {str(e)}")

def process_image(image_path):
    try:
        # Load model and image
        model = load_model()
        image = load_image(image_path)
        image_copy = image.copy()

        # Detect damages
        results, damage_info = detect_damages(image, model)

        # Draw bounding boxes on the image copy
        for damage in damage_info:
            x_min, y_min, x_max, y_max = map(int, damage['bbox'])
            confidence = damage['confidence']
            damage_type = damage['damage_type']
            label = f"{damage_type} ({confidence:.2f})"
            
            # Draw box and label with white background
            cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.rectangle(image_copy, (x_min, y_min - 20), (x_max, y_min), (255, 255, 255), -1)
            cv2.putText(image_copy, label, (x_min, y_min - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Save annotated image
        output_path = image_path.replace('.', '_detected.')
        Image.fromarray(image_copy).save(output_path)

        # Convert damage_info to the expected format while preserving class_id
        detections = [{
            'damage_type': d['damage_type'],
            'confidence': d['confidence'],
            'bbox': d['bbox'].tolist(),
            'class_id': d['class_id']  # Add class_id to the output
        } for d in damage_info]

        return detections, os.path.basename(output_path)

    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}") 
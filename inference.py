import os
from ultralytics import YOLO

model_path = 'models/change_detection_v2/best.pt' 

input_image_path = 'dataset/images/val/149.png' 
output_dir = 'inference_output'

os.makedirs(output_dir, exist_ok=True)

model = YOLO(model_path)
print(f"Model loaded from: {model_path}")

results = model.predict(
    source=input_image_path,
    conf=0.2,
    iou=0.7,
    save=True,
    project=output_dir,
    name='detections'
)

for r in results:
    image_name = os.path.basename(r.path)
    boxes = r.boxes.xywhn # normalized x, y, width, height (center)
    classes = r.boxes.cls # class IDs
    names = r.names # dict of class IDs to names
    confidences = r.boxes.conf # confidence scores

    print(f"\n--- Detections for {image_name} ---")
    if len(boxes) == 0:
        print("  No objects detected.")
    else:
        for i in range(len(boxes)):
            class_id = int(classes[i].item())
            class_name = names[class_id]
            confidence = confidences[i].item()
            bbox = boxes[i].tolist() 

            print(f"  Object: {class_name} (ID: {class_id}), Confidence: {confidence:.2f}, BBox (normalized): {bbox}")

print(f"\nInference complete! Annotated images saved to: {os.path.join(output_dir, 'detections')}")
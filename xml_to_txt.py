import xml.etree.ElementTree as ET
import os
from pathlib import Path

def convert_bbox_to_yolo(size, box):
    """
    Converts PASCAL VOC bounding box coordinates to YOLO format.
    :param size: Tuple of (width, height) of the image.
    :param box: Tuple of (xmin, xmax, ymin, ymax) for the bounding box in pixels.
    :return: Tuple of (x_center, y_center, width, height) normalized from 0 to 1.
    """
    x_center = (box[0] + box[1]) / 2
    y_center = (box[2] + box[3]) / 2
    box_width = box[1] - box[0]
    box_height = box[3] - box[2]

    x_center = x_center / size[0]
    y_center = y_center / size[1]
    box_width = box_width / size[0]
    box_height = box_height / size[1]

    return (x_center, y_center, box_width, box_height)

def convert_xml_to_yolo(xml_filepath, output_dir, class_mapping):
    """
    Parses a PASCAL VOC XML file and converts it to YOLO format (.txt).
    Creates an empty .txt file if no objects are found.
    :param xml_filepath: Path to the input XML file.
    :param output_dir: Directory to save the YOLO .txt files.
    :param class_mapping: Dictionary mapping class names (from XML) to integer IDs (for YOLO).
                          E.g., {'1': 0, '2': 1}
    """
    tree = ET.parse(xml_filepath)
    root = tree.getroot()

    size_element = root.find("size")
    img_width = int(size_element.find("width").text)
    img_height = int(size_element.find("height").text)

    base_filename = Path(xml_filepath).stem
    output_txt_filepath = Path(output_dir) / f"{base_filename}.txt"

    lines = []
    for obj in root.iter("object"):
        class_name = obj.find("name").text
                
        class_id = class_mapping[class_name]

        xmlbox = obj.find("bndbox")
        xmin = float(xmlbox.find("xmin").text)
        ymin = float(xmlbox.find("ymin").text)
        xmax = float(xmlbox.find("xmax").text)
        ymax = float(xmlbox.find("ymax").text)

        bbox = (xmin, xmax, ymin, ymax)
        yolo_coords = convert_bbox_to_yolo((img_width, img_height), bbox)

        lines.append(f"{class_id} {' '.join([str(x) for x in yolo_coords])}")

    with open(output_txt_filepath, "w") as f:
        for line in lines:
            f.write(line + "\n")

    if not lines:
        print(f"Info: No objects found in {xml_filepath}. Creating empty .txt file.")

XML_INPUT_DIR = 'annotations'
YOLO_OUTPUT_DIR = 'labels'
IMAGE_DIR = 'images'

CLASS_MAPPING = {
    '1': 0, # increase
    '2': 1 # decrease
}

os.makedirs(YOLO_OUTPUT_DIR, exist_ok=True)

xml_files = [f for f in os.listdir(XML_INPUT_DIR) if f.endswith('.xml')]
for xml_file in xml_files:
    xml_filepath = Path(XML_INPUT_DIR) / xml_file
    convert_xml_to_yolo(xml_filepath, YOLO_OUTPUT_DIR, CLASS_MAPPING)

image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')]
for image_file in image_files:
    base_filename = Path(image_file).stem
    corresponding_xml = Path(XML_INPUT_DIR) / f"{base_filename}.xml"
    corresponding_yolo_txt = Path(YOLO_OUTPUT_DIR) / f"{base_filename}.txt"
    if not corresponding_xml.exists() and not corresponding_yolo_txt.exists():
        with open(corresponding_yolo_txt, 'w') as f:
            pass
        print(f"Info: Image {image_file} has no XML and no existing YOLO TXT. Created empty TXT file.")

print("\nConversion complete!")
print(f"YOLO .txt annotations saved to: {YOLO_OUTPUT_DIR}")
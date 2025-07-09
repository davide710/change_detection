import os
import cv2

def convert_images_to_gray(image_dir, output_dir):
    """
    Convert all images in the specified directory to grayscale and save them in the output directory.
    
    :param image_dir: Directory containing the original images.
    :param output_dir: Directory where the grayscale images will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_img_path = os.path.join(output_dir, filename)
                cv2.imwrite(gray_img_path, gray_img)
            else:
                print(f"Warning: Could not read image {img_path}. Skipping.")

if __name__ == "__main__":
    IMAGE_DIR = 'images'
    GRAY_OUTPUT_DIR = 'images_gray'
    
    convert_images_to_gray(IMAGE_DIR, GRAY_OUTPUT_DIR)
    print(f"All images converted to grayscale and saved in {GRAY_OUTPUT_DIR}.")
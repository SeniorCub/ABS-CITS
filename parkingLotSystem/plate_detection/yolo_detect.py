# Minimal script: detect license plate with YOLOS and save full + cropped images
import os
import cv2
import argparse
from datetime import datetime
import numpy as np

# Try importing ultralytics YOLO, otherwise exit with helpful message
try:
    from ultralytics import YOLO
except Exception as e:
    raise ImportError('ultralytics YOLO is required. Install with `pip install ultralytics`.') from e


def clamp(v, lo, hi):
    return max(lo, min(int(v), hi))


def detect_and_save(image_path, model_path=None, output_dir='.', conf=0.25):
    """Load image, run YOLOS model, save full image and first detected plate crop (if any).

    Returns tuple: (full_image_path, cropped_image_path_or_None)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f'Image not found: {image_path}')

    if model_path is None:
        # default to the yolos-small model shipped with this repo
        repo_dir = os.path.dirname(__file__)
        model_path = os.path.join(repo_dir, 'yolos-small-rego-plates-detection', 'pytorch_model.bin')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found: {model_path}')

    os.makedirs(output_dir, exist_ok=True)

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError('Failed to read image. Is the path correct and file a valid image?')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    full_filename = f'full_{timestamp}.jpg'
    full_path = os.path.join(output_dir, full_filename)
    cv2.imwrite(full_path, img)

    # Load YOLO model (CPU)
    model = YOLO(model_path)

    # Run inference on the image (force CPU, set confidence)
    results = model(img, conf=conf, device='cpu', verbose=False)

    cropped_path = None

    # Extract boxes from first (and usually only) result
    if results and len(results) > 0:
        result = results[0]
        boxes = getattr(result, 'boxes', None)
        if boxes is not None:
            # Try to get numpy array of xyxy coordinates
            coords = None
            try:
                # ultralytics typically provides boxes.xyxy as a tensor or array
                coords_arr = boxes.xyxy.cpu().numpy()
                if coords_arr.size > 0:
                    coords = coords_arr[0]  # take first detection
            except Exception:
                try:
                    # Fallback if boxes.xyxy is list-like
                    coords_list = list(boxes.xyxy)
                    if len(coords_list) > 0:
                        coords = np.array(coords_list[0])
                except Exception:
                    coords = None

            if coords is not None:
                x1, y1, x2, y2 = coords[:4]
                h, w = img.shape[:2]
                x1c = clamp(x1, 0, w - 1)
                y1c = clamp(y1, 0, h - 1)
                x2c = clamp(x2, 0, w - 1)
                y2c = clamp(y2, 0, h - 1)

                if x2c > x1c and y2c > y1c:
                    crop = img[y1c:y2c, x1c:x2c]
                    crop_filename = f'plate_crop_{timestamp}.jpg'
                    cropped_path = os.path.join(output_dir, crop_filename)
                    cv2.imwrite(cropped_path, crop)

    return full_path, cropped_path


def main():
    parser = argparse.ArgumentParser(description='Detect license plate and save full + cropped images')
    parser.add_argument('--image', '-i', required=True, help='Path to input car image')
    parser.add_argument('--model', '-m', help='Path to YOLOS model file (pytorch_model.bin)')
    parser.add_argument('--out', '-o', default='.', help='Output directory to save images')
    parser.add_argument('--conf', '-c', type=float, default=0.25, help='Detection confidence threshold')

    args = parser.parse_args()

    full, crop = detect_and_save(args.image, model_path=args.model, output_dir=args.out, conf=args.conf)

    print(f'Full image saved: {full}')
    if crop:
        print(f'Cropped plate saved: {crop}')
    else:
        print('No plate detected; only full image was saved.')


if __name__ == '__main__':
    main()

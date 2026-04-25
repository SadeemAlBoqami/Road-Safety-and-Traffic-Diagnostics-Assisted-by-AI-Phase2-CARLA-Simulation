import os
import cv2
import random

DATASET_PATH = 'data'
OUTPUT_PATH = 'quality_check/inspected_samples'

def validate_dataset():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    total_processed = 0

    for i in range(1, 21):
        batch_folder = f'batch_{i}'
        images_dir = os.path.join(DATASET_PATH, 'images', batch_folder)
        labels_dir = os.path.join(DATASET_PATH, 'labels', batch_folder)

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"Skipping {batch_folder}: Directory not found")
            continue

        all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        num_to_sample = min(len(all_images), 15)
        sampled_images = random.sample(all_images, num_to_sample)

        for img_name in sampled_images:
            img_path = os.path.join(images_dir, img_name)
            label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')

            img = cv2.imread(img_path)
            if img is None: continue
            h, w, _ = img.shape

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.split()
                        if len(parts) < 5: continue
                        
                        class_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        
                        x1 = int((cx - bw/2) * w)
                        y1 = int((cy - bh/2) * h)
                        x2 = int((cx + bw/2) * w)
                        y2 = int((cy + bh/2) * h)
                        
                        color = (0, 255, 0) if class_id == 0 else (0, 0, 255)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        
                        label_text = "Vehicle" if class_id == 0 else "Pedestrian"
                        cv2.putText(img, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            output_name = f"{batch_folder}_{img_name}"
            cv2.imwrite(os.path.join(OUTPUT_PATH, output_name), img)
            total_processed += 1
            print(f"Done: {output_name}")

    print(f"\n✅ Finished! Total images in inspected_samples: {total_processed}")

if __name__ == "__main__":
    validate_dataset()
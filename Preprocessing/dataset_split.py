import os
import random
import shutil

# ==========================================
# الإعدادات والمسارات
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "yolo_dataset")
IMAGE_EXT = ".png"
LABEL_EXT = ".txt"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ==========================================
# 1. البحث الشامل بآلية الحماية من التصادم
# ==========================================
def find_all_files():
    images = {}
    labels = {}
    for root, _, files in os.walk(DATASET_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            name, ext = os.path.splitext(file)
            
            # استخراج مسار فرعي لضمان عدم تكرار الأسماء المتشابهة في الدفعات المختلفة
            rel_path = os.path.relpath(root, DATASET_DIR)
            unique_key = os.path.join(rel_path, name)
            
            if ext.lower() == IMAGE_EXT:
                images[unique_key] = file_path
            elif ext.lower() == LABEL_EXT:
                labels[unique_key] = file_path
                
    print(f"Total unique images found: {len(images)}")
    print(f"Total unique labels found: {len(labels)}")
    return images, labels

# ==========================================
# 2. التوأمة الصارمة
# ==========================================
def pair_files(images, labels):
    paired = []
    for unique_key in images:
        if unique_key in labels:
            paired.append((images[unique_key], labels[unique_key]))
            
    print(f"Valid matched pairs: {len(paired)}")
    print(f"Ignored images without labels: {len(images) - len(paired)}")
    print(f"Ignored labels without images: {len(labels) - len(paired)}")
    return paired

# ==========================================
# 3. إنشاء الهيكلية القياسية
# ==========================================
def create_yolo_structure():
    splits = ["train", "val", "test"]
    for split in splits:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

# ==========================================
# 4. التقسيم بالعشوائية الثابتة
# ==========================================
def split_dataset(paired_list):
    random.seed(42)
    random.shuffle(paired_list)
    total = len(paired_list)
    
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    
    train_data = paired_list[:train_end]
    val_data = paired_list[train_end:val_end]
    test_data = paired_list[val_end:]
    
    print(f"\nDataset Split:")
    print(f"Train: {len(train_data)}")
    print(f"Val: {len(val_data)}")
    print(f"Test: {len(test_data)}")
    return train_data, val_data, test_data

# ==========================================
# 5. النسخ الآمن
# ==========================================
def copy_files(data, split_name):
    for img_path, lbl_path in data:
        # إضافة معرّف عشوائي بسيط لتجنب كتابة الملفات فوق بعضها في المجلد النهائي
        # في حال كان هناك صورتان باسم 0000.jpg من مجلدين مختلفين
        base_name = os.path.basename(img_path)
        safe_name = f"{hash(img_path) % 100000}_{base_name}"
        lbl_safe_name = safe_name.replace(IMAGE_EXT, LABEL_EXT)
        
        img_dest = os.path.join(OUTPUT_DIR, "images", split_name, safe_name)
        lbl_dest = os.path.join(OUTPUT_DIR, "labels", split_name, lbl_safe_name)
        
        shutil.copy(img_path, img_dest)
        shutil.copy(lbl_path, lbl_dest)

# ==========================================
# 6. إنشاء ملف data.yaml
# ==========================================
def create_yaml():
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    content = f"""
path: {OUTPUT_DIR}
train: images/train
val: images/val
test: images/test
nc: 2
names: ['Vehicle', 'Pedestrian']
"""
    with open(yaml_path, "w") as f:
        f.write(content.strip())
    print("\ndata.yaml created successfully.")

# ==========================================
# 7. Main Execution
# ==========================================
def main():
    print("Starting dataset preparation...\n")
    images, labels = find_all_files()
    paired_list = pair_files(images, labels)
    
    if len(paired_list) == 0:
        print("No valid pairs found. Exiting.")
        return
        
    create_yolo_structure()
    train_data, val_data, test_data = split_dataset(paired_list)
    
    print("\nCopying training data...")
    copy_files(train_data, "train")
    print("Copying validation data...")
    copy_files(val_data, "val")
    print("Copying test data...")
    copy_files(test_data, "test")
    
    create_yaml()
    print("\nDataset preparation completed successfully!")

if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> ecd2735120b004a9c1a262f54dedbda5c2d9a558

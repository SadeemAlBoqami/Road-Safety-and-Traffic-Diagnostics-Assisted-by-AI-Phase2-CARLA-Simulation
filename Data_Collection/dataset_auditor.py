import os
import glob
import pandas as pd

def audit_dataset():
    # مسارات مجلدات البيانات (تأكدي أنها مطابقة لأسماء المجلدات لديك)
    images_dir = 'dataset/images'
    lstm_dir = 'dataset/lstm'
    
    print("="*40)
    print(" 📊 SYSTEM DATA AUDIT REPORT")
    print("="*40)
    
    # 1. تدقيق بيانات الرؤية (YOLOv8)
    total_images = 0
    batch_folders = glob.glob(os.path.join(images_dir, 'batch_*'))
    
    print("\n[1] Vision Data (YOLOv8 Images):")
    for folder in sorted(batch_folders, key=lambda x: int(os.path.basename(x).split('_')[1])):
        num_images = len(glob.glob(os.path.join(folder, '*.jpg'))) + len(glob.glob(os.path.join(folder, '*.png')))
        print(f" - {os.path.basename(folder).ljust(10)}: {num_images} frames")
        total_images += num_images
        
    print("-" * 30)
    print(f" TOTAL IMAGES: {total_images} / 15000 Target")
    if total_images >= 15000:
        print(" ✅ STATUS: Target Achieved.")
    else:
        print(f" ⚠️ STATUS: Deficit of {15000 - total_images} images.")

    # 2. تدقيق بيانات التنبؤ (LSTM CSVs)
    total_rows = 0
    csv_files = glob.glob(os.path.join(lstm_dir, '*.csv'))
    
    print("\n[2] Temporal Data (LSTM Sequences):")
    for csv_file in sorted(csv_files, key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0])):
        try:
            df = pd.read_csv(csv_file)
            num_rows = len(df)
            print(f" - {os.path.basename(csv_file).ljust(22)}: {num_rows} tracking rows")
            total_rows += num_rows
        except Exception as e:
            print(f" ❌ ERROR reading {os.path.basename(csv_file)}")
            
    print("-" * 30)
    print(f" TOTAL LSTM TRACKING ROWS: {total_rows}")
    print("="*40)

if __name__ == '__main__':
    audit_dataset()
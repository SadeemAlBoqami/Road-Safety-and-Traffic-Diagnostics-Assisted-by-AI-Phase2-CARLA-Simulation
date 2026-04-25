"""
=============================================================
المرحلة 3 - الخطوة أ: إعادة هندسة البيانات وتنظيفها
=============================================================
المدخل  : 20 ملف CSV للحركة + 10 ملفات collision
المخرج  : ملفات numpy جاهزة للتدريب
          X_train/val/test.npy → (N, 30, 6)
          y_train/val/test.npy → (N,)
=============================================================
"""

import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
import os
import glob

# ============================================================
# الإعدادات
# ============================================================
DATASET_DIR  = r"C:\Users\salmr\Desktop\Road-Safety-and-Traffic-Diagnostics-Assisted-by-AI-Phase-2\dataset\lstm"
OUTPUT_DIR = r"C:\Users\salmr\Desktop\Road-Safety-and-Traffic-Diagnostics-Assisted-by-AI-Phase-2\dataset\processed_v2"
WINDOW_SIZE  = 50   # 5 ثوانٍ
STEP_SIZE    = 5    # تداخل جزئي بين السلاسل
PRE_CRASH_FRAMES = 60  # كان 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# معلومات الـ batches
# ============================================================
# batches بدون اصطدام — كل إطاراتها آمنة
SAFE_BATCHES = [2, 4, 5, 9, 14, 16, 17, 18, 19, 20]

# batches فيها اصطدام — لها ملف collision منفصل
COLLISION_BATCHES = [1, 3, 6, 7, 8, 10, 11, 12, 13, 15,
                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30]


# ============================================================
# 1. مرشح كالمان
# ============================================================
def apply_kalman(vx_arr, vy_arr, dt=0.1):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=float)
    kf.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
    kf.R *= 1.0
    kf.Q *= 0.1
    kf.P *= 10.0
    kf.x = np.array([vx_arr[0], vy_arr[0], 0., 0.])

    out = []
    for vx, vy in zip(vx_arr, vy_arr):
        kf.predict()
        kf.update(np.array([vx, vy]))
        out.append(kf.x.copy())
    return np.array(out)  # (T, 4): vx_clean, vy_clean, ax, ay


# ============================================================
# 2. استخلاص الخصائص
# ============================================================
def extract_features(actor_df):
    vx = actor_df["vx"].values.astype(float)
    vy = actor_df["vy"].values.astype(float)

    dvx = np.diff(vx, prepend=vx[0])
    dvy = np.diff(vy, prepend=vy[0])

    filtered  = apply_kalman(vx, vy)
    vx_c, vy_c = filtered[:,0], filtered[:,1]
    ax,   ay   = filtered[:,2], filtered[:,3]

    speed = np.sqrt(vx_c**2 + vy_c**2)
    jerk  = np.gradient(np.sqrt(ax**2 + ay**2))

    return np.stack([dvx, dvy, speed, ax, ay, jerk], axis=1)  # (T, 6)


# ============================================================
# 3. بناء التسميات
# ============================================================
def build_labels(actor_df, collision_frame_id=None, pre_crash=30):
    """
    collision_frame_id: frame_id لحظة الاصطدام من ملف collision CSV
                        None = batch آمنة كلياً
    """
    T      = len(actor_df)
    labels = np.zeros(T, dtype=int)

    if collision_frame_id is not None:
        frames = actor_df["frame_id"].values
        # إيجاد موضع إطار الاصطدام في هذه الvehicles
        collision_pos = np.searchsorted(frames, collision_frame_id)
        collision_pos = min(collision_pos, T - 1)
        start = max(0, collision_pos - pre_crash)
        labels[start:collision_pos + 1] = 1

    return labels


# ============================================================
# 4. تقطيع النافذة الزمنية
# ============================================================
def slice_windows(features, labels, window=30, step=5):
    T = len(features)
    X_list, y_list = [], []
    for start in range(0, T - window + 1, step):
        end   = start + window
        label = int(labels[start:end].max())
        X_list.append(features[start:end])
        y_list.append(label)
    return np.array(X_list), np.array(y_list)


# ============================================================
# 5. Processing batch واحدة
# ============================================================
def process_batch(batch_num, collision_frame_id=None):
    csv_path = os.path.join(DATASET_DIR, f"lstm_data_batch_{batch_num}.csv")
    if not os.path.exists(csv_path):
        print(f"  ⚠️  File not found: {csv_path}")
        return None, None

    df = pd.read_csv(csv_path)
    df = df.sort_values(["actor_id", "frame_id"]).reset_index(drop=True)

    all_X, all_y = [], []

    for actor_id in df["actor_id"].unique():
        actor_df = df[df["actor_id"] == actor_id].reset_index(drop=True)

        if len(actor_df) < WINDOW_SIZE:
            continue

        features = extract_features(actor_df)
        labels   = build_labels(actor_df, collision_frame_id, PRE_CRASH_FRAMES)
        X, y     = slice_windows(features, labels, WINDOW_SIZE, STEP_SIZE)

        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)

    if not all_X:
        return None, None

    return np.concatenate(all_X), np.concatenate(all_y)


# ============================================================
# 6. خط الأنابيب الكامل
# ============================================================
def run_pipeline():
    print("=" * 55)
    print("  Starting Data Pipeline...")
    print("=" * 55)

    all_X, all_y = [], []

    # --- Processing الـ batches الآمنة ---
    print(f"\n📂 Processing safe batches...")
    for batch_num in SAFE_BATCHES:
        X, y = process_batch(batch_num, collision_frame_id=None)
        if X is not None:
            all_X.append(X)
            all_y.append(y)
            print(f"  ✅ Batch {batch_num:2d} → {len(X):5d} sequences | dangerous: {y.sum()}")

    # --- Processing الـ batches ذات الاصطدام ---
    print(f"\n💥 Processing collision batches...")
    for batch_num in COLLISION_BATCHES:
        collision_csv = os.path.join(DATASET_DIR, f"lstm_collision_batch_{batch_num}.csv")
        if not os.path.exists(collision_csv):
            print(f"  ⚠️  Collision file not found for batch {batch_num}")
            continue

        collision_frame_id = int(pd.read_csv(collision_csv)["collision_frame_id"].iloc[0])
        X, y = process_batch(batch_num, collision_frame_id)

        if X is not None:
            all_X.append(X)
            all_y.append(y)
            print(f"  ✅ Batch {batch_num:2d} → {len(X):5d} sequences | dangerous: {y.sum()} | collision_frame: {collision_frame_id}")

    # --- دمج الكل ---
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    print(f"\n📊 Full Dataset Statistics:")
    print(f"   Total sequences  : {len(X_all)}")
    print(f"   Safe      (0)    : {(y_all==0).sum()} ({(y_all==0).mean()*100:.1f}%)")
    print(f"   Dangerous (1)    : {(y_all==1).sum()} ({(y_all==1).mean()*100:.1f}%)")
    print(f"   Array shape      : {X_all.shape}")

    # --- تقسيم عشوائي 80/10/10 ---
    idx       = np.random.permutation(len(X_all))
    t_end     = int(0.8 * len(X_all))
    v_end     = int(0.9 * len(X_all))

    X_train, y_train = X_all[idx[:t_end]],    y_all[idx[:t_end]]
    X_val,   y_val   = X_all[idx[t_end:v_end]], y_all[idx[t_end:v_end]]
    X_test,  y_test  = X_all[idx[v_end:]],     y_all[idx[v_end:]]

    # --- تطبيع Z-Score (إحصاءات التدريب فقط) ---
    mean = X_train.mean(axis=(0,1), keepdims=True)
    std  = X_train.std(axis=(0,1),  keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std
    X_test  = (X_test  - mean) / std

    # --- حفظ ---
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"),    X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"),      X_val)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"),     X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"),    y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"),      y_val)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),     y_test)
    np.save(os.path.join(OUTPUT_DIR, "norm_mean.npy"),  mean)
    np.save(os.path.join(OUTPUT_DIR, "norm_std.npy"),   std)

    print(f"\n✅ Saved to: {OUTPUT_DIR}")
    print(f"   X_train : {X_train.shape} | y_train : {y_train.shape}")
    print(f"   X_val   : {X_val.shape}   | y_val   : {y_val.shape}")
    print(f"   X_test  : {X_test.shape}  | y_test  : {y_test.shape}")
    print("=" * 55)


if __name__ == "__main__":
    run_pipeline()
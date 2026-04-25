import os
import csv
import cv2


IMAGE_DIR = "dataset/images"
LABEL_DIR = "dataset/labels"
LSTM_DIR = "dataset/lstm"


os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(LSTM_DIR, exist_ok=True)


# ============================
# حفظ بيانات YOLO
# ============================

def save_yolo(frame_counter, image_data, bboxes, batch_number: int | None = None):
    frame_id = f"{frame_counter:06d}"

    if batch_number is not None:
        image_base = os.path.join(IMAGE_DIR, f"batch_{batch_number}")
        label_base = os.path.join(LABEL_DIR, f"batch_{batch_number}")
        os.makedirs(image_base, exist_ok=True)
        os.makedirs(label_base, exist_ok=True)
        image_path = os.path.join(image_base, f"{frame_id}.png")
        label_path = os.path.join(label_base, f"{frame_id}.txt")
    else:
        image_path = os.path.join(IMAGE_DIR, f"{frame_id}.png")
        label_path = os.path.join(LABEL_DIR, f"{frame_id}.txt")

    # حفظ الصورة
    cv2.imwrite(image_path, image_data)

    # حفظ bounding boxes
    with open(label_path, "w") as f:
        for bbox in bboxes:
            class_id = bbox["class_id"]
            x_center = bbox["x_center"]
            y_center = bbox["y_center"]
            width = bbox["width"]
            height = bbox["height"]

            f.write(
                f"{class_id} "
                f"{x_center} "
                f"{y_center} "
                f"{width} "
                f"{height}\n"
            )


# ============================
# حفظ بيانات LSTM
# ============================

def save_lstm_csv(frame_counter, actors, batch_number: int | None = None):
    if batch_number is not None:
        csv_path = os.path.join(LSTM_DIR, f"lstm_data_batch_{batch_number}.csv")
    else:
        csv_path = os.path.join(LSTM_DIR, "trajectory.csv")

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)

        # كتابة header
        if not file_exists:
            writer.writerow(
                [
                    "frame_id",
                    "actor_id",
                    "x",
                    "y",
                    "vx",
                    "vy",
                ]
            )

        for actor in actors:
            try:
                actor_id = actor.id

                location = actor.get_location()
                velocity = actor.get_velocity()

                writer.writerow(
                    [
                        frame_counter,
                        actor_id,
                        location.x,
                        location.y,
                        velocity.x,
                        velocity.y,
                    ]
                )
            except Exception:
                continue

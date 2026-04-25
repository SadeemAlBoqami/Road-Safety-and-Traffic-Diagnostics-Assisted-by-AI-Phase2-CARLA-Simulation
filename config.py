import carla

# image dimensions for YOLO input
IMAGE_SIZE = (640, 640)

# LSTM Frame Rate (Hz)
FPS = 10

# Total number of images to collect
TOTAL_IMAGES = 15000 

# Camera position relative to the ego vehicle
CAM_POSITION = carla.Transform(
    carla.Location(x=1.5, z=2.4), 
    carla.Rotation(pitch=0, yaw=0, roll=0)
)

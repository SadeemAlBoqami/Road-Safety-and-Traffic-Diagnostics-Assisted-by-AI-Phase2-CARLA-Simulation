import numpy as np
import carla
import config

def get_2d_boxes(world, camera, image_data):
    bboxes = []
    # التأكد من ترتيب الأبعاد (العرض ثم الطول)
    image_w, image_h = config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]
    
    # مصفوفة الكاميرا
    fov = 90.0
    f = image_w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = f
    K[0, 2] = image_w / 2.0
    K[1, 2] = image_h / 2.0

    # مصفوفة التحويل من العالم إلى الكاميرا
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

    for npc in world.get_actors().filter('vehicle.*'):
        if npc.id == camera.parent.id:
            continue

        # فلتر المسافة 50 متر
        dist = npc.get_transform().location.distance(camera.get_transform().location)
        if dist > 50:
            continue

        box = npc.bounding_box
        verts = [v for v in box.get_world_vertices(npc.get_transform())]
        x_coords, y_coords = [], []
        
        for v in verts:
            p_world = np.array([v.x, v.y, v.z, 1.0])
            p_camera_carla = np.dot(world_2_camera, p_world)
            
            # --- التعديل الجذري: تبديل المحاور لتتوافق مع نظام الرؤية الحاسوبية ---
            # [X_cv, Y_cv, Z_cv] = [Y_carla, -Z_carla, X_carla]
            p_camera_cv = np.array([
                p_camera_carla[1], 
                -p_camera_carla[2], 
                p_camera_carla[0]
            ])
            
            # التحقق من أن النقطة أمام الكاميرا (العمق Z أصبح الآن هو الأمام)
            if p_camera_cv[2] > 0:
                p_pixel = np.dot(K, p_camera_cv)
                # التجانس (Homogenization)
                p_pixel /= p_pixel[2]
                x_coords.append(p_pixel[0])
                y_coords.append(p_pixel[1])

        # رسم المربع فقط إذا كانت هناك نقاط صالحة على الشاشة
        if len(x_coords) > 0 and len(y_coords) > 0:
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # قص المربع ليبقى داخل حدود الصورة (Clipping)
            min_x = max(0, min_x)
            max_x = min(image_w, max_x)
            min_y = max(0, min_y)
            max_y = min(image_h, max_y)

            # حساب قيم YOLO (تتراوح بين 0 و 1)
            cx = ((min_x + max_x) / 2.0) / image_w
            cy = ((min_y + max_y) / 2.0) / image_h
            w = (max_x - min_x) / image_w
            h = (max_y - min_y) / image_h
            
            # تصفية المربعات غير المنطقية
            if 0 < cx < 1 and 0 < cy < 1 and w > 0 and h > 0:
                bboxes.append({
                    "class_id": 0,  # 0 للسيارات
                    "x_center": cx,
                    "y_center": cy,
                    "width": w,
                    "height": h
                })
                
    return bboxes
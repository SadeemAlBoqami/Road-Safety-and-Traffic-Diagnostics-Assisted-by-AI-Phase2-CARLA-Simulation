import random
import queue
from typing import List
import math

import carla
import numpy as np
import cv2

import config
from modules import camera_sensor, projection, traffic_manager, data_writer


# رقم الدُفعة (Batch) — غيّره يدويًا قبل التشغيل لتفادي تداخل البيانات بين التشغيلات
BATCH_NUMBER = 20
# عدد الإطارات المطلوب جمعها في كل تشغيل (يجب أن يكون 1000 حسب المتطلبات)
FRAMES_PER_BATCH = 1000


# تفعيل وضع المزامنة (Synchronous) لضمان جمع بيانات ثابتة عند 10 FPS
# نُعيد الإعدادات السابقة حتى نرجعها في cleanup
def _enable_synchronous_mode(world: carla.World) -> carla.WorldSettings:
    previous_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / config.FPS
    world.apply_settings(settings)
    return previous_settings


# Green Wave: تثبيت جميع إشارات المرور على الأخضر لمنع الاختناقات/التوقف
def _apply_green_wave(world: carla.World) -> None:
    traffic_lights = world.get_actors().filter("traffic.traffic_light*")
    for tl in traffic_lights:
        try:
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)
        except RuntimeError:
            continue


def _set_episode_weather(world: carla.World) -> None:
    # للباتش 16 و 17 (المطر):
    world.set_weather(carla.WeatherParameters.ClearNight)


# إنشاء سيارة الـ Ego (Tesla Model 3) في نقطة Spawn عشوائية
def _spawn_ego_vehicle(world: carla.World, blueprint_library: carla.BlueprintLibrary) -> carla.Actor:
    tesla_bp_candidates = [
        bp
        for bp in blueprint_library.filter("vehicle.tesla.*")
        if bp.id.endswith("model3") or "model3" in bp.id
    ]

    vehicle_bp = tesla_bp_candidates[0] if tesla_bp_candidates else blueprint_library.find(
        "vehicle.tesla.model3"
    )

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available for ego vehicle.")

    ego_transform = random.choice(spawn_points)
    ego_vehicle = world.spawn_actor(vehicle_bp, ego_transform)
    if ego_vehicle is None:
        raise RuntimeError("Failed to spawn ego vehicle.")

    return ego_vehicle


# توليد حركة مرور كثيفة: 120 مركبة بالضبط (مع إدخال 2‑Wheelers إن وُجدت)
# إذا لم نتمكن من الوصول للعدد المطلوب نرمي RuntimeError لضمان "Exactly"
def _spawn_traffic_vehicles(
    world: carla.World,
    blueprint_library: carla.BlueprintLibrary,
    ego_vehicle: carla.Actor,
    total_vehicles: int,
) -> List[carla.Actor]:
    spawn_points = list(world.get_map().get_spawn_points())
    if not spawn_points:
        raise RuntimeError("No spawn points available for traffic vehicles.")

    vehicle_bps = list(blueprint_library.filter("vehicle.*"))
    if not vehicle_bps:
        raise RuntimeError("No vehicle blueprints found.")

    two_wheel_bps: List[carla.ActorBlueprint] = []
    other_bps: List[carla.ActorBlueprint] = []
    for bp in vehicle_bps:
        try:
            wheels = int(bp.get_attribute("number_of_wheels").as_int())
            if wheels == 2:
                two_wheel_bps.append(bp)
            else:
                other_bps.append(bp)
        except Exception:
            other_bps.append(bp)

    desired_two_wheel = 0
    if two_wheel_bps:
        desired_two_wheel = max(1, min(12, total_vehicles // 10))

    spawned: List[carla.Actor] = []
    remaining_two_wheel = desired_two_wheel
    remaining_other = total_vehicles - desired_two_wheel

    random.shuffle(spawn_points)
    attempts = 0
    max_attempts = total_vehicles * 80

    while len(spawned) < total_vehicles and attempts < max_attempts:
        spawn_transform = spawn_points[attempts % len(spawn_points)]

        if remaining_two_wheel > 0 and two_wheel_bps:
            bp = random.choice(two_wheel_bps)
        else:
            bp = random.choice(other_bps if other_bps else vehicle_bps)

        actor = world.try_spawn_actor(bp, spawn_transform)
        attempts += 1

        if actor is None:
            continue
        if actor.id == ego_vehicle.id:
            try:
                actor.destroy()
            except RuntimeError:
                pass
            continue

        spawned.append(actor)
        if remaining_two_wheel > 0 and two_wheel_bps:
            try:
                wheels = int(bp.get_attribute("number_of_wheels").as_int())
                if wheels == 2:
                    remaining_two_wheel -= 1
                else:
                    remaining_other -= 1
            except Exception:
                remaining_other -= 1
        else:
            remaining_other -= 1

    if len(spawned) != total_vehicles:
        raise RuntimeError(f"Could not spawn exactly {total_vehicles} traffic vehicles (spawned {len(spawned)}).")

    return spawned


# توليد مشاة: 80 مشاة يمشون بالضبط + وحدات التحكم الخاصة بهم (Controllers)
# نُرجع الاثنين معًا لتسهيل التدمير الكامل في cleanup
def _spawn_pedestrians(world: carla.World, blueprint_library: carla.BlueprintLibrary, count: int) -> List[carla.Actor]:
    walker_bps = blueprint_library.filter("walker.pedestrian.*")
    controller_bp = blueprint_library.find("controller.ai.walker")

    if not walker_bps:
        raise RuntimeError("No pedestrian blueprints found.")

    walkers: List[carla.Actor] = []
    controllers: List[carla.Actor] = []

    attempts = 0
    max_attempts = count * 120

    while len(walkers) < count and attempts < max_attempts:
        location = world.get_random_location_from_navigation()
        attempts += 1
        if location is None:
            continue

        walker_transform = carla.Transform(location)
        walker_bp = random.choice(list(walker_bps))

        walker = world.try_spawn_actor(walker_bp, walker_transform)
        if walker is None:
            continue

        controller = world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
        if controller is None:
            try:
                walker.destroy()
            except RuntimeError:
                pass
            continue

        controllers.append(controller)
        walkers.append(walker)

    if len(walkers) != count:
        raise RuntimeError(f"Could not spawn exactly {count} pedestrians (spawned {len(walkers)}).")

    # Start pedestrian movement
    for controller in controllers:
        try:
            controller.start()
            controller.go_to_location(world.get_random_location_from_navigation())
            controller.set_max_speed(1.5)
        except RuntimeError:
            continue

    # Return walkers and controllers together for unified cleanup
    return walkers + controllers


# إعداد Traffic Manager:
# - تشغيل Autopilot لكل مركبات NPC
# - منح الـ Ego "مناعة" وتعديلات مضادة للتوقف (anti-stuck)
def _configure_traffic_manager(
    client: carla.Client,
    ego_vehicle: carla.Actor,
    traffic_vehicles: List[carla.Actor],
) -> carla.TrafficManager:
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    # Configure all spawned vehicles
    for vehicle in traffic_vehicles:
        try:
            vehicle.set_autopilot(True, tm.get_port())
        except RuntimeError:
            continue

    # Ego vehicle: full immunity + anti-stuck tuning
    ego_vehicle.set_autopilot(True, tm.get_port())
    tm.ignore_lights_percentage(ego_vehicle, 100.0)
    tm.ignore_signs_percentage(ego_vehicle, 100.0)
    tm.ignore_vehicles_percentage(ego_vehicle, 0.0)
    tm.ignore_walkers_percentage(ego_vehicle, 100.0)
    tm.force_lane_change(ego_vehicle, False)
    tm.auto_lane_change(ego_vehicle, True)

    # Stuck-vehicle fixes
    tm.distance_to_leading_vehicle(ego_vehicle, 2.5)
    tm.set_desired_speed(ego_vehicle, 25.0)

    return tm


# إضافة كاميرا RGB على سيارة الـ Ego وإرسال آخر إطار فقط إلى Queue (لتفادي تراكم الذاكرة)
def _attach_camera(
    world: carla.World,
    ego_vehicle: carla.Actor,
    sensor_queue: "queue.Queue[carla.Image]",
) -> carla.Sensor:
    camera_bp = camera_sensor.setup_blueprint(world)
    camera = world.spawn_actor(camera_bp, config.CAM_POSITION, attach_to=ego_vehicle)
    if camera is None:
        raise RuntimeError("Failed to spawn camera sensor.")

    def _put_latest(image: carla.Image) -> None:
        try:
            if sensor_queue.full():
                sensor_queue.get_nowait()
            sensor_queue.put_nowait(image)
        except queue.Empty:
            pass
        except queue.Full:
            pass

    camera.listen(_put_latest)
    return camera


# تحويل صورة CARLA الخام إلى مصفوفة OpenCV (BGR) ثم تغيير الحجم إلى config.IMAGE_SIZE
def _carla_image_to_bgr_array(image: carla.Image) -> np.ndarray:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    bgr = array[:, :, :3][:, :, ::-1]
    resized = cv2.resize(bgr, config.IMAGE_SIZE)
    return resized


#
# نقطة التشغيل الرئيسية:
# - اتصال CARLA
# - تحميل خريطة عشوائية (Town01 أو Town03 فقط عبر traffic_manager.select_town)
# - تشغيل المزامنة + الطقس + Green Wave
# - Spawn: Ego + 120 مركبة + 80 مشاة
# - جمع 1000 إطار وحفظ YOLO + CSV بحسب رقم الدفعة
# - Cleanup صارم في finally لتفادي تسريب الذاكرة
def main() -> None:
    # الاتصال بالمحاكي
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    # تحميل عالم/خريطة عشوائية لهذا التشغيل
    world = client.load_world(traffic_manager.select_town())

    # حفظ حالة الإعدادات السابقة للرجوع لها عند الانتهاء
    world_settings_backup: carla.WorldSettings | None = None
    # قائمة بكل العناصر التي يجب تدميرها (كاميرا/سيارات/مشاة/Controllers)
    actors_to_destroy: List[carla.Actor] = []
    # حساس الكاميرا (نحتاجه لإيقاف الاستماع قبل destroy)
    camera: carla.Sensor | None = None
    # Traffic Manager (نطفئ وضع المزامنة الخاص به في النهاية)
    tm: carla.TrafficManager | None = None

    try:
        # إعدادات البيئة (مزامنة + طقس + Green Wave)
        world_settings_backup = _enable_synchronous_mode(world)
        _set_episode_weather(world)
        _apply_green_wave(world)

        # مكتبة الـ Blueprints لاختيار أنواع المركبات/المشاة
        blueprint_library = world.get_blueprint_library()

        # Spawn سيارة الـ Ego
        ego_vehicle = _spawn_ego_vehicle(world, blueprint_library)
        actors_to_destroy.append(ego_vehicle)

        # Spawn 120 مركبة فوضوية
        traffic_vehicles = _spawn_traffic_vehicles(
            world=world,
            blueprint_library=blueprint_library,
            ego_vehicle=ego_vehicle,
            total_vehicles=80,
        )
        actors_to_destroy.extend(traffic_vehicles)

        # Spawn 80 مشاة + Controllers
        pedestrians = _spawn_pedestrians(
            world=world,
            blueprint_library=blueprint_library,
            count=80,
        )
        actors_to_destroy.extend(pedestrians)

        # إعداد Traffic Manager + تفعيل إعدادات منع توقف الـ Ego
        tm = _configure_traffic_manager(
            client=client,
            ego_vehicle=ego_vehicle,
            traffic_vehicles=traffic_vehicles,
        )

        # Queue صغيرة للاحتفاظ بأحدث صورة فقط (تجنّب تراكم الإطارات)
        sensor_queue: "queue.Queue[carla.Image]" = queue.Queue(maxsize=1)
        # تركيب الكاميرا على الـ Ego
        camera = _attach_camera(world, ego_vehicle, sensor_queue)
        actors_to_destroy.append(camera)

        # تهيئة أول Tick لضمان بدء وصول بيانات الحساسات
        world.tick()

        # عدّاد الإطارات داخل هذه الدُفعة
        frame_id = 0
        # عدّاد لاكتشاف التوقف (Deadlock): عدد الإطارات المتتالية التي تكون فيها سرعة الـ Ego أقل من 1 كم/س
        stuck_frames = 0

        # حلقة جمع البيانات: 1000 إطار بالضبط
        while frame_id < FRAMES_PER_BATCH:
            # Tick واحد = إطار واحد في الوضع المتزامن
            world.tick()

            try:
                # استلام أحدث صورة من الكاميرا
                image_data: carla.Image = sensor_queue.get(timeout=2.0)
            except queue.Empty:
                continue

            # تحويل الصورة لصيغة OpenCV
            rgb_image = _carla_image_to_bgr_array(image_data)
            # استخراج مربعات 2D (YOLO) باستخدام projection.py
            bboxes = projection.get_2d_boxes(world, camera, image_data)

            # حفظ الصورة والـ labels داخل مجلد الدُفعة
            data_writer.save_yolo(frame_id, rgb_image, bboxes, batch_number=BATCH_NUMBER)

            # حفظ بيانات الـ LSTM في ملف CSV خاص بالدُفعة لمنع overwrite
            actors_snapshot = world.get_actors().filter("vehicle.*")
            data_writer.save_lstm_csv(frame_id, actors_snapshot, batch_number=BATCH_NUMBER)

            # حساب سرعة سيارة الـ Ego (كم/س) لاكتشاف التوقف أو الاصطدام
            ego_velocity = ego_vehicle.get_velocity()
            speed_m_s = math.sqrt(
                ego_velocity.x * ego_velocity.x
                + ego_velocity.y * ego_velocity.y
                + ego_velocity.z * ego_velocity.z
            )
            speed_km_h = speed_m_s * 3.6

            if speed_km_h < 1.0:
                stuck_frames += 1
            else:
                stuck_frames = 0

            # Fail-Fast: إذا كانت السيارة متوقفة أكثر من 50 إطارًا متتاليًا نعتبر أن هناك Deadlock
            if stuck_frames > 50:
                raise RuntimeError(
                    "Deadlock detected! Batch corrupted. Please delete this batch folder and re-run."
                )

            # الانتقال للإطار التالي
            frame_id += 1

    finally:
        # إطفاء مزامنة Traffic Manager (احتياطي)
        if tm is not None:
            try:
                tm.set_synchronous_mode(False)
            except RuntimeError:
                pass

        # إعادة إعدادات العالم كما كانت قبل التعديل
        if world_settings_backup is not None:
            try:
                world.apply_settings(world_settings_backup)
            except RuntimeError:
                pass

        # إيقاف الكاميرا قبل تدميرها لتفادي أخطاء وقت الإغلاق
        if camera is not None:
            try:
                camera.stop()
            except RuntimeError:
                pass

        # تنظيف صارم لكل العناصر لمنع تسريب الذاكرة بين التشغيلات
        for actor in reversed(actors_to_destroy):
            if actor is None:
                continue
            try:
                actor.destroy()
            except RuntimeError:
                continue


if __name__ == "__main__":
    # تشغيل السكربت
    main()
import carla
import random

def select_town():
    towns = ["Town01", "Town03"]
    return random.choice(towns)

def set_random_weather(world):
    probability = random.random()

    if probability < 0.4:
        weather = carla.WeatherParameters.ClearNoon
    elif probability < 0.6:
        weather = carla.WeatherParameters.WetCloudyNoon
    else:
        weather = carla.WeatherParameters.SoftRainSunset

    world.set_weather(weather)

# --- التعديل هنا فقط: إضافة ego_vehicle في المدخلات ---
def setup_traffic_manager(client, world, ego_vehicle):
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    vehicles = world.get_actors().filter("vehicle.*")

    for vehicle in vehicles:
        # --- التعديل هنا: استثناء سيارة المستخدم من القيادة العشوائية للمرور ---
        if vehicle.id != ego_vehicle.id:
            vehicle.set_autopilot(True, traffic_manager.get_port())

            if random.random() < 0.5:
                traffic_manager.ignore_lights_percentage(vehicle, 100)
                traffic_manager.ignore_signs_percentage(vehicle, 100)

    return traffic_manager
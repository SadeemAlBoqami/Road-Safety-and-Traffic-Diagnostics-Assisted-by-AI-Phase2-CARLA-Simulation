import carla
import config  

def setup_blueprint(world):
    
    
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')

    
    camera_bp.set_attribute('image_size_x', str(config.IMAGE_SIZE[0]))
    camera_bp.set_attribute('image_size_y', str(config.IMAGE_SIZE[1]))
    
    
    camera_bp.set_attribute('sensor_tick', str(1.0 / config.FPS))

    return camera_bp

def spawn_camera(world, ego_vehicle, sensor_queue):
    
    camera_bp = setup_blueprint(world)
    
    spawn_point = config.CAM_POSITION
    
    camera = world.spawn_actor(
        camera_bp, 
        spawn_point, 
        attach_to=ego_vehicle
    )
    
    camera.listen(sensor_queue.put) 
    
    return camera

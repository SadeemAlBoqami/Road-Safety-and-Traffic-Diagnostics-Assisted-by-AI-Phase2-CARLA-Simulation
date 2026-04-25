"""
=============================================================
LSTM Data Collection v2 — Diverse Collision Scenarios
=============================================================
New vs original:
  1. Four scenario types with controlled speeds
  2. Slower collisions = longer pre-crash window
  3. Batch numbers 21-30 (no overwrite of existing data)
  4. Each scenario tuned for 200-500 frames before collision

Scenarios:
  REAR_END    : Ego follows closely at medium speed
  CUT_IN      : Traffic vehicle cuts in front of Ego
  INTERSECTION: Two vehicles converge at junction
  HARD_BRAKE  : Leading vehicle brakes suddenly
=============================================================
"""

import random
import queue
import threading
import math
from typing import List
from enum import Enum

import carla
import numpy as np
import pandas as pd
import os

import config
from modules import traffic_manager, data_writer

# ============================================================
# Scenario Types
# ============================================================
class Scenario(Enum):
    REAR_END     = "rear_end"
    CUT_IN       = "cut_in"
    INTERSECTION = "intersection"
    HARD_BRAKE   = "hard_brake"

# ============================================================
# Settings
# ============================================================
OUTPUT_DIR       = r"C:\Users\salmr\Desktop\Road-Safety-and-Traffic-Diagnostics-Assisted-by-AI-Phase-2\dataset\lstm"
FRAMES_PER_BATCH = 1000
BATCH_NUMBER     = 21  # Start from 21


# ============================================================
# Helpers (unchanged from v1)
# ============================================================
def _enable_synchronous_mode(world):
    previous = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / config.FPS
    world.apply_settings(settings)
    return previous


def _apply_green_wave(world):
    for tl in world.get_actors().filter("traffic.traffic_light*"):
        try:
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)
        except RuntimeError:
            continue

def _set_episode_weather(world: carla.World, weather: carla.WeatherParameters) -> None:
    world.set_weather(weather)


def _spawn_ego_vehicle(world, blueprint_library):
    candidates = [
        bp for bp in blueprint_library.filter("vehicle.tesla.*")
        if bp.id.endswith("model3") or "model3" in bp.id
    ]
    bp = candidates[0] if candidates else blueprint_library.find("vehicle.tesla.model3")
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available.")
    ego = world.spawn_actor(bp, random.choice(spawn_points))
    if ego is None:
        raise RuntimeError("Failed to spawn ego vehicle.")
    return ego


def _spawn_traffic_vehicles(world, blueprint_library, ego_vehicle, total_vehicles):
    spawn_points = list(world.get_map().get_spawn_points())
    vehicle_bps  = list(blueprint_library.filter("vehicle.*"))
    spawned, attempts = [], 0
    random.shuffle(spawn_points)
    while len(spawned) < total_vehicles and attempts < total_vehicles * 80:
        bp    = random.choice(vehicle_bps)
        actor = world.try_spawn_actor(bp, spawn_points[attempts % len(spawn_points)])
        attempts += 1
        if actor is None or actor.id == ego_vehicle.id:
            continue
        spawned.append(actor)
    if len(spawned) != total_vehicles:
        raise RuntimeError(f"Could not spawn {total_vehicles} vehicles (got {len(spawned)}).")
    return spawned


def _spawn_pedestrians(world, blueprint_library, count):
    walker_bps    = blueprint_library.filter("walker.pedestrian.*")
    controller_bp = blueprint_library.find("controller.ai.walker")
    walkers, controllers = [], []
    attempts = 0
    while len(walkers) < count and attempts < count * 120:
        loc = world.get_random_location_from_navigation()
        attempts += 1
        if loc is None:
            continue
        walker = world.try_spawn_actor(
            random.choice(list(walker_bps)), carla.Transform(loc)
        )
        if walker is None:
            continue
        ctrl = world.try_spawn_actor(
            controller_bp, carla.Transform(), attach_to=walker
        )
        if ctrl is None:
            walker.destroy()
            continue
        walkers.append(walker)
        controllers.append(ctrl)
    if len(walkers) != count:
        raise RuntimeError(f"Could not spawn {count} pedestrians.")
    for ctrl in controllers:
        try:
            ctrl.start()
            ctrl.go_to_location(world.get_random_location_from_navigation())
            ctrl.set_max_speed(1.4)
        except RuntimeError:
            continue
    return walkers + controllers


def _attach_collision_sensor(world, ego_vehicle, collision_event, collision_frame_holder):
    bp     = world.get_blueprint_library().find("sensor.other.collision")
    sensor = world.spawn_actor(bp, carla.Transform(), attach_to=ego_vehicle)

    def _on_collision(event):
        if not collision_event.is_set():
            collision_frame_holder[0] = event.frame
            collision_event.set()

    sensor.listen(_on_collision)
    return sensor


def _save_collision_record(batch_number, collision_frame_id, output_dir):
    path = os.path.join(output_dir, f"lstm_collision_batch_{batch_number}.csv")
    pd.DataFrame({"collision_frame_id": [collision_frame_id]}).to_csv(path, index=False)
    print(f"  💥 Collision saved: frame={collision_frame_id} → {path}")


# ============================================================
# Core: Scenario-based Traffic Manager Configuration
# ============================================================
def _configure_for_scenario(client, ego_vehicle, traffic_vehicles, scenario: Scenario):
    """
    Each scenario configures the Traffic Manager differently
    to produce realistic, slow-building collision patterns.
    """
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    # All traffic vehicles use autopilot
    for v in traffic_vehicles:
        try:
            v.set_autopilot(True, tm.get_port())
        except RuntimeError:
            continue

    ego_vehicle.set_autopilot(True, tm.get_port())

    # --- Always ignore lights and signs ---
    tm.ignore_lights_percentage(ego_vehicle, 100.0)
    tm.ignore_signs_percentage(ego_vehicle, 100.0)

    if scenario == Scenario.REAR_END:
        # ─────────────────────────────────────────────────────
        # Rear-End: Ego follows a vehicle closely at medium speed
        # Expected collision: 200-400 frames
        # Signal pattern: gradual dvx decrease (closing gap)
        # ─────────────────────────────────────────────────────
        print("  📌 Scenario: REAR_END (gradual approach)")
        tm.ignore_walkers_percentage(ego_vehicle, 100.0)
        tm.ignore_vehicles_percentage(ego_vehicle, 30.0)  # ← كان 0.0
        tm.distance_to_leading_vehicle(ego_vehicle, 0.5)  # ← كان 1.0
        tm.set_desired_speed(ego_vehicle, 40.0)            # ← كان 35.0
        tm.vehicle_percentage_speed_difference(ego_vehicle, -30.0)  # ← كان -20.0

        for v in traffic_vehicles[:20]:
            try:
               tm.set_desired_speed(v, 15.0)              # ← كان 20.0
               tm.distance_to_leading_vehicle(v, 3.0)
            except RuntimeError:
                continue

    elif scenario == Scenario.CUT_IN:
        # ─────────────────────────────────────────────────────
        # Cut-In: Traffic vehicles change lanes aggressively
        # Expected collision: 100-300 frames
        # Signal pattern: sudden dvy spike then dvx increase
        # ─────────────────────────────────────────────────────
        print("  📌 Scenario: CUT_IN (aggressive lane change)")
        tm.ignore_walkers_percentage(ego_vehicle, 100.0)
        tm.ignore_vehicles_percentage(ego_vehicle, 50.0)  # ← partially ignores
        tm.distance_to_leading_vehicle(ego_vehicle, 2.0)
        tm.set_desired_speed(ego_vehicle, 40.0)

        # Traffic vehicles: force frequent lane changes
        for v in traffic_vehicles:
            try:
                tm.auto_lane_change(v, True)
                tm.vehicle_percentage_speed_difference(v, random.uniform(-40, 10))
                tm.distance_to_leading_vehicle(v, random.uniform(0.5, 2.0))
            except RuntimeError:
                continue

    elif scenario == Scenario.INTERSECTION:
        # ─────────────────────────────────────────────────────
        # Intersection: Ego runs red lights at junctions
        # Expected collision: 150-400 frames
        # Signal pattern: perpendicular velocity vectors converging
        # ─────────────────────────────────────────────────────
        print("  📌 Scenario: INTERSECTION (junction conflict)")
        tm.ignore_walkers_percentage(ego_vehicle, 100.0)
        tm.ignore_vehicles_percentage(ego_vehicle, 100.0)
        tm.ignore_lights_percentage(ego_vehicle, 100.0)   # run all red lights
        tm.distance_to_leading_vehicle(ego_vehicle, 0.5)
        tm.set_desired_speed(ego_vehicle, 30.0)           # ← slower at junctions
        tm.vehicle_percentage_speed_difference(ego_vehicle, -10.0)

        # Traffic: normal speed to create crossing conflicts
        for v in traffic_vehicles:
            try:
                tm.ignore_lights_percentage(v, 80.0)      # most also run lights
                tm.set_desired_speed(v, 25.0)
            except RuntimeError:
                continue

    elif scenario == Scenario.HARD_BRAKE:
        # ─────────────────────────────────────────────────────
        # Hard Brake: Leading vehicle decelerates sharply
        # Expected collision: 200-500 frames
        # Signal pattern: strong negative ax (jerk spike)
        # ─────────────────────────────────────────────────────
        print("  📌 Scenario: HARD_BRAKE (sudden deceleration)")
        tm.ignore_walkers_percentage(ego_vehicle, 100.0)
        tm.ignore_vehicles_percentage(ego_vehicle, 0.0)   # ← follows vehicles
        tm.distance_to_leading_vehicle(ego_vehicle, 1.5)  # ← close following
        tm.set_desired_speed(ego_vehicle, 45.0)
        tm.vehicle_percentage_speed_difference(ego_vehicle, -30.0)

        # Leading vehicles: some stop suddenly
        for i, v in enumerate(traffic_vehicles[:15]):
            try:
                if i % 3 == 0:
                    tm.set_desired_speed(v, 5.0)          # ← near-stop
                    tm.distance_to_leading_vehicle(v, 0.1)
                else:
                    tm.set_desired_speed(v, 30.0)
            except RuntimeError:
                continue

    return tm


# ============================================================
# Main batch runner
# ============================================================
def run_single_batch(
    batch_number: int,
    scenario: Scenario,
    weather: carla.WeatherParameters,
    total_vehicles: int,
    output_dir: str,
    frames_per_batch: int = FRAMES_PER_BATCH,
) -> dict:

    client = carla.Client("localhost", 2000)
    client.set_timeout(15.0)
    world  = client.load_world(traffic_manager.select_town())

    world_settings_backup  = None
    actors_to_destroy      = []
    collision_sensor_actor = None
    tm                     = None
    collision_frame_holder = [None]
    collision_event        = threading.Event()

    result = {
        "batch"             : batch_number,
        "scenario"          : scenario.value,
        "frames_collected"  : 0,
        "collision_occurred": False,
        "collision_frame_id": None,
    }

    try:
        world_settings_backup = _enable_synchronous_mode(world)
        _set_episode_weather(world, weather)
        _apply_green_wave(world)

        bp_lib = world.get_blueprint_library()

        ego_vehicle = _spawn_ego_vehicle(world, bp_lib)
        actors_to_destroy.append(ego_vehicle)

        traffic_vehicles = _spawn_traffic_vehicles(
            world, bp_lib, ego_vehicle, total_vehicles
        )
        actors_to_destroy.extend(traffic_vehicles)

        pedestrians = _spawn_pedestrians(world, bp_lib, count=60)
        actors_to_destroy.extend(pedestrians)

        # Configure scenario-specific TM
        tm = _configure_for_scenario(client, ego_vehicle, traffic_vehicles, scenario)

        collision_sensor_actor = _attach_collision_sensor(
            world, ego_vehicle, collision_event, collision_frame_holder
        )
        actors_to_destroy.append(collision_sensor_actor)

        world.tick()

        frame_id    = 0
        stuck_count = 0

        while frame_id < frames_per_batch:

            if collision_event.is_set():
                print(f"  ⚠️  Collision at frame {frame_id}")
                result["collision_occurred"]   = True
                result["collision_frame_id"]   = frame_id
                break

            world.tick()

            actors_snapshot = world.get_actors().filter("vehicle.*")
            data_writer.save_lstm_csv(
                frame_id, actors_snapshot, batch_number=batch_number
            )

            v         = ego_vehicle.get_velocity()
            speed_kmh = math.sqrt(v.x**2 + v.y**2 + v.z**2) * 3.6
            stuck_count = stuck_count + 1 if speed_kmh < 0.5 else 0

            if stuck_count > 80:
                raise RuntimeError("Deadlock — batch corrupted.")

            frame_id += 1

        result["frames_collected"] = frame_id

        if result["collision_occurred"] and result["collision_frame_id"] is not None:
            _save_collision_record(
                batch_number, result["collision_frame_id"], output_dir
            )

    finally:
        if tm:
            try: tm.set_synchronous_mode(False)
            except RuntimeError: pass
        if world_settings_backup:
            try: world.apply_settings(world_settings_backup)
            except RuntimeError: pass
        if collision_sensor_actor:
            try: collision_sensor_actor.stop()
            except RuntimeError: pass
        for actor in reversed(actors_to_destroy):
            if actor is None: continue
            try: actor.destroy()
            except RuntimeError: continue

    return result


# ============================================================
# Entry point — run manually, change BATCH_NUMBER each time
# ============================================================
if __name__ == "__main__":

    # ── Edit these two lines before each run ──
    BATCH_NUMBER = 30
    SCENARIO     = Scenario.HARD_BRAKE
    # ──────────────────────────────────────────

    SCENARIO_WEATHER = {
        Scenario.REAR_END    : carla.WeatherParameters(cloudiness=40),
        Scenario.CUT_IN      : carla.WeatherParameters.ClearNoon,
        Scenario.INTERSECTION: carla.WeatherParameters(cloudiness=60),
        Scenario.HARD_BRAKE  : carla.WeatherParameters.ClearNoon,
    }

    result = run_single_batch(
        batch_number   = BATCH_NUMBER,
        scenario       = SCENARIO,
        weather        = SCENARIO_WEATHER[SCENARIO],
        total_vehicles = 120,
        output_dir     = OUTPUT_DIR,
    )
    print(f"\n✅ Result: {result}")

    # Force save collision if occurred but file missing
if result.get("collision_occurred") and result.get("collision_frame_id"):
    _save_collision_record(
        result["batch"],
        result["collision_frame_id"],
        OUTPUT_DIR
    )
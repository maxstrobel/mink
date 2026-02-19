"""Benchmark IK performance on realistic trajectories.

Measures end-to-end per-step wall time: target updates + IK solve + integration.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import mujoco
import numpy as np

import mink
from mink.lie import SE3

_HERE = Path(__file__).parent
_G1_XML = _HERE.parent / "examples" / "unitree_g1" / "scene.xml"
_ALOHA_XML = _HERE.parent / "examples" / "aloha" / "scene.xml"

_ALOHA_JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]


# ---------------------------------------------------------------------------
# Humanoid G1
# ---------------------------------------------------------------------------


def setup_humanoid():
    model = mujoco.MjModel.from_xml_path(_G1_XML.as_posix())
    configuration = mink.Configuration(model)
    feet = ["right_foot", "left_foot"]

    tasks = [
        pelvis_task := mink.FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=10.0,
            lm_damping=1e-3,
        ),
        torso_task := mink.FrameTask(
            frame_name="torso_link",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=5.0,
            lm_damping=1e-3,
        ),
        posture_task := mink.PostureTask(model, cost=1e-1),
        com_task := mink.ComTask(cost=50.0),
    ]

    feet_tasks = []
    for foot in feet:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="site",
            position_cost=100.0,
            orientation_cost=10.0,
            lm_damping=0.0,
        )
        feet_tasks.append(task)
    tasks.extend(feet_tasks)

    r_hand_task = mink.FrameTask(
        frame_name="right_palm",
        frame_type="site",
        position_cost=5.0,
        orientation_cost=0.5,
        gain=0.5,
        lm_damping=1e-3,
    )
    tasks.append(r_hand_task)

    relative_task = mink.RelativeFrameTask(
        frame_name="left_palm",
        frame_type="site",
        root_name="torso_link",
        root_type="body",
        position_cost=5.0,
        orientation_cost=0.5,
        gain=0.5,
        lm_damping=1e-3,
    )
    tasks.append(relative_task)

    collision_pairs = [
        (["left_hand_collision"], ["left_thigh_collision"]),
        (["right_hand_collision"], ["right_thigh_collision"]),
    ]
    collision_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=collision_pairs,
        minimum_distance_from_collisions=0.005,
        collision_detection_distance=0.15,
    )
    limits = [mink.ConfigurationLimit(model), collision_limit]

    com_mid = model.body("com_target").mocapid[0]
    feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in feet]
    r_hand_mid = model.body("right_palm_target").mocapid[0]

    model = configuration.model
    data = configuration.data

    configuration.update_from_keyframe("stand")
    posture_task.set_target_from_configuration(configuration)
    pelvis_task.set_target_from_configuration(configuration)
    torso_task.set_target_from_configuration(configuration)
    for foot in feet:
        mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
    mink.move_mocap_to_frame(model, data, "right_palm_target", "right_palm", "site")
    data.mocap_pos[com_mid] = data.subtree_com[1].copy()
    relative_task.set_target_from_configuration(configuration)

    return {
        "model": model,
        "data": data,
        "configuration": configuration,
        "tasks": tasks,
        "limits": limits,
        "damping": 1e-2,
        "com_task": com_task,
        "feet_tasks": feet_tasks,
        "feet_mid": feet_mid,
        "r_hand_task": r_hand_task,
        "r_hand_mid": r_hand_mid,
        "relative_task": relative_task,
        "init_relative_target": relative_task.transform_target_to_root,
        "com_mid": com_mid,
        "init_com": data.subtree_com[1].copy(),
        "init_r_hand_pos": data.mocap_pos[r_hand_mid].copy(),
    }


def step_humanoid(s, t, dt):
    data = s["data"]

    squat_phase = 0.5 * (1.0 - np.cos(2.0 * np.pi * 0.5 * t))
    com_target = s["init_com"].copy()
    com_target[2] -= 0.12 * squat_phase
    data.mocap_pos[s["com_mid"]] = com_target

    angle = 2.0 * np.pi * 0.8 * t
    r_pos = s["init_r_hand_pos"].copy()
    r_pos[0] += 0.04 * np.cos(angle)
    r_pos[2] += 0.04 * np.sin(angle)
    data.mocap_pos[s["r_hand_mid"]] = r_pos

    wave_t = 2.0 * np.pi * 0.8 * t
    dy = 0.05 + 0.04 * np.sin(wave_t)
    dz = 0.04 * np.cos(wave_t)
    target_in_root = s["init_relative_target"] @ SE3.from_translation(
        np.array([0.0, dy, dz])
    )
    s["relative_task"].set_target(target_in_root)

    s["com_task"].set_target(data.mocap_pos[s["com_mid"]])
    for i, ft in enumerate(s["feet_tasks"]):
        ft.set_target(mink.SE3.from_mocap_id(data, s["feet_mid"][i]))
    s["r_hand_task"].set_target(mink.SE3.from_mocap_id(data, s["r_hand_mid"]))

    vel = mink.solve_ik(
        s["configuration"],
        s["tasks"],
        dt,
        "daqp",
        damping=s["damping"],
        limits=s["limits"],
    )
    s["configuration"].integrate_inplace(vel, dt)


# ---------------------------------------------------------------------------
# ALOHA dual-arm
# ---------------------------------------------------------------------------


def setup_aloha():
    model = mujoco.MjModel.from_xml_path(str(_ALOHA_XML))
    data = mujoco.MjData(model)

    joint_names = []
    velocity_limits = {}
    for prefix in ["left", "right"]:
        for n in _ALOHA_JOINT_NAMES:
            name = f"{prefix}/{n}"
            joint_names.append(name)
            velocity_limits[name] = np.pi

    configuration = mink.Configuration(model)

    tasks = [
        l_ee_task := mink.FrameTask(
            frame_name="left/gripper",
            frame_type="site",
            position_cost=5.0,
            orientation_cost=1.0,
            gain=1.0,
            lm_damping=1e-3,
        ),
        r_ee_task := mink.FrameTask(
            frame_name="right/gripper",
            frame_type="site",
            position_cost=5.0,
            orientation_cost=1.0,
            gain=1.0,
            lm_damping=1e-3,
        ),
    ]

    l_wrist = mink.get_subtree_geom_ids(model, model.body("left/wrist_link").id)
    r_wrist = mink.get_subtree_geom_ids(model, model.body("right/wrist_link").id)
    l_geoms = mink.get_subtree_geom_ids(model, model.body("left/upper_arm_link").id)
    r_geoms = mink.get_subtree_geom_ids(model, model.body("right/upper_arm_link").id)
    frame_geoms = mink.get_body_geom_ids(model, model.body("metal_frame").id)
    collision_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=[(l_wrist, r_wrist), (l_geoms + r_geoms, frame_geoms + ["table"])],
        minimum_distance_from_collisions=0.05,
        collision_detection_distance=0.1,
    )

    limits = [
        mink.ConfigurationLimit(model=model),
        mink.VelocityLimit(model, velocity_limits),
        collision_limit,
    ]

    l_mid = model.body("left/target").mocapid[0]
    r_mid = model.body("right/target").mocapid[0]

    mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
    configuration.update(data.qpos)
    mujoco.mj_forward(model, data)
    mink.move_mocap_to_frame(model, data, "left/target", "left/gripper", "site")
    mink.move_mocap_to_frame(model, data, "right/target", "right/gripper", "site")

    return {
        "model": model,
        "data": data,
        "configuration": configuration,
        "tasks": tasks,
        "limits": limits,
        "damping": 1e-4,
        "l_ee_task": l_ee_task,
        "r_ee_task": r_ee_task,
        "l_mid": l_mid,
        "r_mid": r_mid,
        "init_l_pos": data.mocap_pos[l_mid].copy(),
        "init_r_pos": data.mocap_pos[r_mid].copy(),
        "init_l_quat": data.mocap_quat[l_mid].copy(),
        "init_r_quat": data.mocap_quat[r_mid].copy(),
    }


def step_aloha(s, t, dt):
    data = s["data"]
    phase = 2.0 * np.pi * 0.4 * t

    inward = 0.5 * (1.0 - np.cos(phase))
    x_sweep = 0.03 + (0.16 - 0.03) * inward
    z_drop = -0.24 * inward
    y_push = -0.04 * (1.0 - inward)

    l_pos = s["init_l_pos"].copy()
    l_pos[0] += x_sweep
    l_pos[1] += y_push
    l_pos[2] += z_drop

    r_pos = s["init_r_pos"].copy()
    r_pos[0] -= x_sweep
    r_pos[1] += y_push
    r_pos[2] += z_drop

    data.mocap_pos[s["l_mid"]] = l_pos
    data.mocap_pos[s["r_mid"]] = r_pos
    data.mocap_quat[s["l_mid"]] = s["init_l_quat"]
    data.mocap_quat[s["r_mid"]] = s["init_r_quat"]

    s["l_ee_task"].set_target(mink.SE3.from_mocap_id(data, s["l_mid"]))
    s["r_ee_task"].set_target(mink.SE3.from_mocap_id(data, s["r_mid"]))

    vel = mink.solve_ik(
        s["configuration"],
        s["tasks"],
        dt,
        "daqp",
        damping=s["damping"],
        limits=s["limits"],
    )
    s["configuration"].integrate_inplace(vel, dt)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

SCENARIOS = {
    "humanoid": (
        "Humanoid G1 (49-DOF, collision avoidance)",
        setup_humanoid,
        step_humanoid,
    ),
    "aloha": (
        "ALOHA dual-arm (1104 geom pairs, collision avoidance)",
        setup_aloha,
        step_aloha,
    ),
}


def run_benchmark(scenario: str, n_steps: int, n_warmup: int, dt: float = 1.0 / 500.0):
    _, setup_fn, step_fn = SCENARIOS[scenario]
    s = setup_fn()
    step_times = []
    t = 0.0
    for i in range(n_warmup + n_steps):
        t0 = time.perf_counter()
        step_fn(s, t, dt)
        elapsed = time.perf_counter() - t0
        if i >= n_warmup:
            step_times.append(elapsed * 1e6)
        t += dt
    return step_times


def compute_stats(times: list[float]) -> dict:
    s = sorted(times)
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "p95": s[int(0.95 * len(s))],
        "p99": s[int(0.99 * len(s))],
        "std": statistics.stdev(times),
        "min": s[0],
        "max": s[-1],
    }


def print_stats(label: str, stats: dict):
    print(f"\n  {label}")
    for k in ("mean", "median", "p95", "p99", "std", "min", "max"):
        print(f"  {k:>8s}: {stats[k]:8.1f} us")


def print_comparison(stats_a: dict, stats_b: dict, label_a: str, label_b: str):
    print(f"\n  {'Metric':<8s}  {label_a:>14s}  {label_b:>14s}  {'Speedup':>8s}")
    print(f"  {'-' * 8}  {'-' * 14}  {'-' * 14}  {'-' * 8}")
    for metric in ("mean", "median", "p95", "p99"):
        a, b = stats_a[metric], stats_b[metric]
        speedup = b / a if a > 0 else float("inf")
        print(f"  {metric:<8s}  {a:14.1f}  {b:14.1f}  {speedup:7.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark IK on realistic trajectories",
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        choices=list(SCENARIOS.keys()),
        help="Scenario to benchmark",
    )
    parser.add_argument("--save", type=str, help="Save results to JSON")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("A", "B"),
        help="Compare two saved JSON files (prints A vs B speedup)",
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=500)
    args = parser.parse_args()

    if args.compare:
        with open(args.compare[0]) as f:
            a = json.load(f)
        with open(args.compare[1]) as f:
            b = json.load(f)
        la = Path(args.compare[0]).stem
        lb = Path(args.compare[1]).stem
        print()
        print("=" * 58)
        print("  Trajectory Benchmark — Branch Comparison")
        print("=" * 58)
        print_comparison(b, a, lb, la)
        print()
        return

    if not args.scenario:
        parser.error("scenario is required when not using --compare")

    label, _, _ = SCENARIOS[args.scenario]
    print()
    print("=" * 58)
    print(f"  {label}")
    print(f"  {args.steps} steps (after {args.warmup} warmup)")
    print("=" * 58)

    times = run_benchmark(args.scenario, args.steps, args.warmup)
    stats = compute_stats(times)
    print_stats("Results", stats)
    print()

    if args.save:
        with open(args.save, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved to {args.save}")
        print()


if __name__ == "__main__":
    main()

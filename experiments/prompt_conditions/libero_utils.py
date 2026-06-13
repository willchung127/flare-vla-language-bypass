"""Shared helpers for the VLA language-bypass experiments.

Observation construction mirrors openpi/examples/libero/main.py exactly so that
results are comparable to standard pi0 LIBERO evals.
"""

import math
import os
import random

import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

LIBERO_DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
NUM_SETTLE_STEPS = 10  # let objects settle after set_init_state (same as openpi example)

MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}

# Rule-based paraphrases: weak on purpose. LIBERO-Pro shows even minimal
# rephrasing breaks VLAs, so if these already hurt, the point is made.
# Order matters; only the first match per instruction is applied for variant 1.
_SYNONYMS = [
    ("turn on", "switch on"),
    ("pick up", "grab"),
    ("on top of", "onto"),
    ("put", "place"),
    ("open", "pull open"),
    ("push", "slide"),
    ("close", "shut"),
]


def make_paraphrases(instr: str) -> list[str]:
    """Two paraphrases: (1) synonym substitution, (2) politeness prefix."""
    p1 = instr
    for a, b in _SYNONYMS:
        if a in p1:
            p1 = p1.replace(a, b, 1)
            break
    p2 = "please " + instr
    return [p1, p2]


def scramble(instr: str, seed: int = 0) -> str:
    words = instr.split()
    rng = random.Random(seed)
    rng.shuffle(words)
    return " ".join(words)


def quat2axisangle(quat):
    """Copied from robosuite (same as openpi libero example)."""
    quat = np.array(quat, dtype=np.float64)
    quat[3] = float(np.clip(quat[3], -1.0, 1.0))
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def build_element(obs, prompt: str, resize: int = 224):
    """LIBERO obs dict -> openpi policy input element (matches examples/libero/main.py)."""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize, resize))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize, resize))
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )
    return {
        "observation/image": img,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": str(prompt),
    }


def get_suite(suite_name: str):
    return benchmark.get_benchmark_dict()[suite_name]()


def make_env(task, seed: int = 0, img_size: int = 256):
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=img_size, camera_widths=img_size
    )
    env.seed(seed)
    return env


def reset_to_init_state(env, task_suite, task_id: int, init_idx: int):
    env.reset()
    init_states = task_suite.get_task_init_states(task_id)
    obs = env.set_init_state(init_states[init_idx % len(init_states)])
    for _ in range(NUM_SETTLE_STEPS):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    return obs

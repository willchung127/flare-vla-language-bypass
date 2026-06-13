"""Multimode method matrix on LIBERO-Long.

6 conditions:
  1. ODE-single           — baseline
  2. SDE-single (η=0.2)   — confirms SDE-alone null
  3. ODE + BoN K=8 + V3   — selection-side TTC
  4. SDE + BoN K=8 + V3   — does SDE add to BoN?
  5. Guided-ODE + V3      — gradient guidance alone (no exploration noise)
  6. Guided-SDE + V3      — gradient + exploration noise (DPS/Langevin)

Scale: 5 tasks × 10 trials × 6 conds = 300 episodes.
  conds 1, 2, 5, 6: K=1 NFE each
  conds 3, 4: K=8 NFE each
  Wall time estimate: (50 × 0.3 + 50 × 0.3 + 50 × 2.4 + 50 × 2.4 + 50 × 0.4 + 50 × 0.4) min ≈ 13 GPU-hr

Usage:
    cd ~/flare/openpi && \\
    PYTHONUNBUFFERED=1 TORCH_COMPILE_DISABLE=1 \\
      uv run python -u ~/flare/remote_multimode_matrix.py \\
      --conditions 1 2 3 4 5 6 --n-tasks 5 --n-trials 10 \\
      --alpha 0.1 --eta 0.2
"""
from __future__ import annotations
import argparse
import collections
import json
import math
import os
import re
import sys
import time
from pathlib import Path


def read_bddl_language(bddl_path: str) -> str | None:
    """Extract the `(:language ...)` string from a BDDL file.

    Returns the literal text inside the parens (without the closing paren),
    or None if no language section is found.
    """
    text = Path(bddl_path).read_text()
    m = re.search(r"\(:language\s+([^)]+?)\)", text)
    if m:
        return m.group(1).strip()
    return None

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, str(Path.home() / "flare"))

import numpy as np
import torch
from openpi_client import image_tools
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi.training import config as _c
from openpi.policies import policy_config
from patch_pi0_sde import apply_patch
from verifiers.score import build_scorers
from verifiers.models import V3MC, WMTerminal

try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False


# =============================================================================
# Config
# =============================================================================
TASK_SUITE = "libero_10"
REPLAN_STEPS = 5
BASE_SEED = 7000
CHECKPOINT_DIR = str(Path.home() / "flare/checkpoints/pi0_libero_pt")
OUT_DIR = Path.home() / "flare/results/multimode_matrix"
LIBERO_DUMMY_ACTION = np.array([0., 0., 0., 0., 0., 0., -1.], dtype=np.float32)
NUM_STEPS_WAIT = 10
MAX_STEPS_MAP = {"libero_10": 520, "libero_90": 520,
                 "libero_spatial": 220, "libero_object": 280,
                 "libero_goal": 300, "libero_long": 520}

V3MC_CKPT = str(Path.home() / "flare/results/v3_mc_state.pt")
WMT_CKPT = str(Path.home() / "flare/results/wm_terminal_state.pt")
V2_CACHE = str(Path.home() / "flare/results/v2_outcome_cache.npz")


# =============================================================================
# Conditions
# =============================================================================
CONDITIONS = {
    1: {"name": "ODE_single",       "eta": 0.0, "K": 1, "guidance": False, "verifier": None,        "guide_verif": None},
    2: {"name": "SDE_single",       "eta": 0.2, "K": 1, "guidance": False, "verifier": None,        "guide_verif": None},
    3: {"name": "ODE_BoN_V3",       "eta": 0.0, "K": 8, "guidance": False, "verifier": "V3_MC",     "guide_verif": None},
    4: {"name": "SDE_BoN_V3",       "eta": 0.2, "K": 8, "guidance": False, "verifier": "V3_MC",     "guide_verif": None},
    5: {"name": "GuidedODE_V3",     "eta": 0.0, "K": 1, "guidance": True,  "verifier": None,        "guide_verif": "V3_MC"},
    6: {"name": "GuidedSDE_V3",     "eta": 0.2, "K": 1, "guidance": True,  "verifier": None,        "guide_verif": "V3_MC"},
    # Verifier-generalization ablations
    7: {"name": "GuidedSDE_WMT",    "eta": 0.2, "K": 1, "guidance": True,  "verifier": None,        "guide_verif": "WM_terminal"},
    8: {"name": "ODE_BoN_WMT",      "eta": 0.0, "K": 8, "guidance": False, "verifier": "WM_terminal","guide_verif": None},
    # Composition (Day 13): guidance + BoN selection
    9: {"name": "GuidedSDE_BoN4_V3","eta": 0.2, "K": 4, "guidance": True,  "verifier": "V3_MC",     "guide_verif": "V3_MC"},
    # NFE-scaling: ODE BoN at different K values for Pareto plot
    10: {"name": "ODE_BoN2_V3",     "eta": 0.0, "K": 2, "guidance": False, "verifier": "V3_MC",     "guide_verif": None},
    11: {"name": "ODE_BoN4_V3",     "eta": 0.0, "K": 4, "guidance": False, "verifier": "V3_MC",     "guide_verif": None},
    # Composition K-scaling (Day 14): composition at K=2 and K=8
    12: {"name": "GuidedSDE_BoN2_V3","eta": 0.2, "K": 2, "guidance": True, "verifier": "V3_MC",     "guide_verif": "V3_MC"},
    13: {"name": "GuidedSDE_BoN8_V3","eta": 0.2, "K": 8, "guidance": True, "verifier": "V3_MC",     "guide_verif": "V3_MC"},
    # Composition verifier-generalization (Day 14): WMT in BoN role with V3 guidance
    14: {"name": "GuidedSDE_BoN4_WMT","eta": 0.2,"K": 4, "guidance": True, "verifier": "WM_terminal","guide_verif": "V3_MC"},
    # Multi-mode quicktest (Day 16): SDE+BoN at K=4 to match composition's NFE
    15: {"name": "SDE_BoN4_V3",      "eta": 0.2, "K": 4, "guidance": False, "verifier": "V3_MC",     "guide_verif": None},
}


# Per-task object-name pairs for Plan A (mode-B exemplar BoN).
# Tuple is (mode_A_object, mode_B_object) — the object the policy currently
# prefers vs the one we want to recover at inference.
TASK_MODE_OBJECTS = {
    0: ("alphabet_soup_1_pos", "tomato_sauce_1_pos"),
    1: ("cream_cheese_1_pos", "butter_1_pos"),
    # 7, 8 to be added if/when we extend the experiment to them
}


def mode_b_score_batch(
    candidates: np.ndarray,         # (K, 50, 7)
    current_eef_pos: np.ndarray,    # (3,)
    mode_a_obj_pos: np.ndarray,     # (3,)
    mode_b_obj_pos: np.ndarray,     # (3,)
    num_actions: int = 20,
) -> np.ndarray:
    """Score K candidate chunks by how mode-B-aligned their early actions are.

    Plan A scorer: pi0 outputs delta-eef commands in the first 3 action dims
    (LIBERO uses OSC_POSE control). Summing the first `num_actions` of those
    deltas gives a rough heading vector — where would the gripper end up if
    we executed those actions? We compare this heading vector to:
        v_A = mode_A_object - current_eef
        v_B = mode_B_object - current_eef
    and score each candidate by (cos sim to v_B) - (cos sim to v_A).

    Higher score = chunk points more toward mode B than mode A. BoN-select max.
    No training data required; uses only known object positions from the scene.
    """
    K = candidates.shape[0]
    scores = np.zeros(K, dtype=np.float64)

    to_a = mode_a_obj_pos - current_eef_pos
    to_b = mode_b_obj_pos - current_eef_pos
    a_norm = float(np.linalg.norm(to_a) + 1e-6)
    b_norm = float(np.linalg.norm(to_b) + 1e-6)

    for k in range(K):
        chunk = candidates[k]
        # Sum first num_actions delta-eef commands to estimate net heading
        direction = chunk[:num_actions, :3].sum(axis=0)
        d_norm = float(np.linalg.norm(direction) + 1e-6)
        cos_a = float(np.dot(direction, to_a) / (d_norm * a_norm))
        cos_b = float(np.dot(direction, to_b) / (d_norm * b_norm))
        scores[k] = cos_b - cos_a

    return scores


def _quat2axisangle(q):
    q = np.array(q, dtype=np.float64)
    q[3] = float(np.clip(q[3], -1.0, 1.0))
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((q[:3] * 2.0 * math.acos(q[3])) / den).astype(np.float32)


def format_obs(o, prompt, sz=224):
    img = np.ascontiguousarray(o["agentview_image"][::-1, ::-1])
    wimg = np.ascontiguousarray(o["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, sz, sz))
    wimg = image_tools.convert_to_uint8(image_tools.resize_with_pad(wimg, sz, sz))
    state = np.concatenate([
        o["robot0_eef_pos"].astype(np.float32),
        _quat2axisangle(o["robot0_eef_quat"]),
        o["robot0_gripper_qpos"].astype(np.float32),
    ])
    return {"observation/image": img, "observation/wrist_image": wimg,
            "observation/state": state, "prompt": prompt}


# =============================================================================
# Verifier wrapper — converts V3-MC into a callable usable inside the SDE patch
# =============================================================================
class V3MCGuidanceCallable:
    """Wraps V3-MC for in-integration use.

    Called with (obs_features_tensor: (B, 8), chunk_tensor: (B, H_act, D)) inside
    a torch.enable_grad context. Returns scalar logit per batch element.

    The SDE patch will autograd this with respect to chunk_tensor to compute the
    guidance gradient.
    """
    def __init__(self, ckpt_path: str, device: str):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model = V3MC(
            obs_dim=ckpt["obs_dim"],
            chunk_dim=ckpt["chunk_dim"],
            hidden=ckpt["config"]["hidden"],
            dropout=0.0,
        ).to(device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.device = device
        # V3-MC was trained with flattened chunks (350-d for pi0's 50*7)
        self.chunk_repr = ckpt["chunk_repr"]
        assert self.chunk_repr == "flattened", \
            f"V3-MC was trained with {self.chunk_repr}, guidance expects 'flattened'"

    def __call__(self, obs_features: torch.Tensor, chunk: torch.Tensor) -> torch.Tensor:
        """Returns (B,) logits. chunk shape (B, H_act, D_act). obs_features (B, 8).

        pi0 internally uses D_act=32 (model's max action dim, padded). LIBERO
        actually uses only first 7 dims (6-DoF delta-pose + gripper). V3-MC was
        trained on 7-d chunks → slice before flattening.
        """
        if chunk.dim() == 3 and chunk.shape[-1] > 7:
            chunk = chunk[..., :7]  # (B, H_act, 7)
        chunk_flat = chunk.reshape(chunk.shape[0], -1)  # (B, 50*7 = 350)
        return self.model(obs_features, chunk_flat)


class WMTGuidanceCallable:
    """Wraps WMT (terminal-state predictor) for in-integration use.

    Score = -L2(predicted_terminal, task_centroid) / per_dim_std
    Higher score = closer to known-success terminal for this task = better.
    Differentiable: gradient pulls chunks toward predicting terminals closer
    to the task's success centroid.

    Requires set_task(task_id) before each rollout to know which centroid.
    """
    def __init__(self, ckpt_path: str, device: str):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model = WMTerminal(
            obs_dim=ckpt["obs_dim"],
            chunk_dim=ckpt["chunk_dim"],
            hidden=ckpt["config"]["hidden"],
            dropout=0.0,
        ).to(device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.device = device
        self.chunk_repr = ckpt["chunk_repr"]
        # Per-task target centroids (4-d: eef_x, eef_y, eef_z, gripper_open)
        self.task_targets = {
            int(k): torch.tensor(v, dtype=torch.float32, device=device)
            for k, v in ckpt["task_targets"].items()
        }
        self.label_std = torch.tensor(ckpt["label_std"], dtype=torch.float32, device=device)
        self.current_task_id = 0

    def set_task(self, task_id: int):
        """Update current task — required before each rollout."""
        self.current_task_id = int(task_id)

    def __call__(self, obs_features: torch.Tensor, chunk: torch.Tensor) -> torch.Tensor:
        if chunk.dim() == 3 and chunk.shape[-1] > 7:
            chunk = chunk[..., :7]
        chunk_flat = chunk.reshape(chunk.shape[0], -1)
        pred = self.model(obs_features, chunk_flat)  # (B, 4)
        target = self.task_targets.get(self.current_task_id,
                                        torch.zeros(4, device=self.device))
        scaled_diff = (pred - target.unsqueeze(0)) / (self.label_std.unsqueeze(0) + 1e-6)
        # -L2 → higher score = closer = better
        return -torch.norm(scaled_diff, dim=-1)


# =============================================================================
# Rollout — handles all 6 condition variants by setting model attributes
# =============================================================================
def rollout(policy, model, task, task_id: int, bddl, init_state, env_seed: int,
            base_seed: int, cond: dict, scorer_fn, guidance_callable,
            obs_state_for_guidance, replan_steps: int = REPLAN_STEPS,
            video_mode: str = "off", frame_stride: int = 2,
            language_override: str | None = None,
            modeb_bon_config: dict | None = None,
            hdcspi_subtasks: list | None = None):
    """One closed-loop episode with the given condition.

    For BoN modes: sample K candidates via policy.infer (which uses SDE patch),
      then score with scorer_fn and execute best.
    For single modes: just call policy.infer once.
    For guided modes: set model.flare_verifier_fn and model.flare_alpha; call
      policy.infer once (the SDE patch applies guidance internally).
    """
    # Set model attributes for sampler control (used by SDE patch)
    model.flare_eta = cond["eta"]
    if cond["guidance"]:
        model.flare_verifier_fn = guidance_callable
        model.flare_alpha = cond.get("alpha", 0.1)
        model.flare_obs_state = obs_state_for_guidance  # the 8-d state for V3-MC
        # WMT-based guidance: set the per-task target
        if hasattr(guidance_callable, "set_task"):
            guidance_callable.set_task(task_id)
    else:
        model.flare_verifier_fn = None
        model.flare_alpha = 0.0

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(env_seed)
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    # Record initial absolute object positions for post-hoc first-moved analysis
    # (filters out `*_to_*_pos` relative-coord keys which contaminate the signal)
    def _abs_obj_pos(o):
        out = {}
        for k, v in o.items():
            if k.endswith("_pos") and not k.startswith("robot") and "_to_" not in k:
                try:
                    arr = np.asarray(v, dtype=np.float64).flatten()
                    if arr.size >= 3:
                        out[k] = arr[:3].tolist()
                except Exception:
                    pass
        return out
    initial_object_positions = _abs_obj_pos(obs)
    waypoints = []   # (t, object_positions) every WAYPOINT_EVERY steps
    WAYPOINT_EVERY = 25
    frames = [] if video_mode != "off" else None

    # Resolve mode-B BoN config (Plan A) — look up object positions for this task
    modeb_active = False
    modeb_num_actions = 20
    mode_a_obj_pos = mode_b_obj_pos = None
    if modeb_bon_config is not None:
        modeb_num_actions = int(modeb_bon_config.get("deadreckon_n", 20))
        if task_id not in TASK_MODE_OBJECTS:
            print(f"  [modeb-bon] WARN: no mode mapping for task_id={task_id}; "
                  f"falling back to default scorer", flush=True)
        else:
            ma_name, mb_name = TASK_MODE_OBJECTS[task_id]
            if ma_name in initial_object_positions and mb_name in initial_object_positions:
                mode_a_obj_pos = np.asarray(initial_object_positions[ma_name][:3], dtype=np.float64)
                mode_b_obj_pos = np.asarray(initial_object_positions[mb_name][:3], dtype=np.float64)
                modeb_active = True
            else:
                missing = [n for n in (ma_name, mb_name) if n not in initial_object_positions]
                print(f"  [modeb-bon] WARN: missing initial positions for {missing}; "
                      f"keys present: {list(initial_object_positions.keys())[:8]}...", flush=True)

    action_plan = collections.deque()
    done = False
    t = 0
    chunk_counter = 0
    # Use task suite from the env (which is set per-rollout by caller); fall back to default
    max_steps = MAX_STEPS_MAP.get(getattr(rollout, "_task_suite", TASK_SUITE), 520)
    K = cond["K"]

    effective_language = language_override if language_override is not None else task.language

    # HDCSPI setup: track which subtask we're on, override effective_language with subtask prompt
    hdcspi_idx = 0
    hdcspi_log = []  # records when each subtask transition happened
    if hdcspi_subtasks:
        effective_language = hdcspi_subtasks[0]["prompt"]
        print(f"  [HDCSPI] starting subtask 0: prompt='{effective_language}', "
              f"target='{hdcspi_subtasks[0]['target_object']}'", flush=True)

    # If directed-noise is active, prepare for per-chunk bias direction computation
    directed_noise_active = (
        getattr(model, "flare_noise_bias_strength", 0.0) > 0
        and task_id in TASK_MODE_OBJECTS
    )
    if directed_noise_active:
        # Mode-B object name lookup
        _, _mb_name = TASK_MODE_OBJECTS[task_id]

    while t < max_steps and not done:
        if not action_plan:
            example = format_obs(obs, effective_language)
            # Update obs_state for guidance (in case it changed across replan)
            model.flare_obs_state = torch.from_numpy(example["observation/state"]).float()

            # Set per-chunk directed-noise bias direction (Track 2)
            if directed_noise_active:
                current_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)[:3]
                if _mb_name in obs:
                    target_pos = np.asarray(obs[_mb_name], dtype=np.float64)[:3]
                    direction = target_pos - current_eef
                    dnorm = float(np.linalg.norm(direction) + 1e-6)
                    direction_unit = (direction / dnorm).astype(np.float32)  # (3,)
                    # Build chunk-space bias: first 3 dims (delta-eef) get direction,
                    # remaining 4 dims (orientation, gripper) stay zero
                    bias = np.zeros((50, 7), dtype=np.float32)
                    bias[:, :3] = direction_unit  # broadcast over 50 timesteps
                    model.flare_noise_bias_direction = torch.from_numpy(bias).float()

            if K == 1:
                # Single sample (with or without guidance, handled by patch internally)
                seed_k = base_seed + 10000 * chunk_counter
                np.random.seed(seed_k); torch.manual_seed(seed_k)
                chunk = policy.infer(example)["actions"]
            else:
                # BoN: K candidates, then verifier-select
                candidates = []
                for k in range(K):
                    seed_k = base_seed + 10000 * chunk_counter + 137 * k
                    np.random.seed(seed_k); torch.manual_seed(seed_k)
                    c = policy.infer(example)["actions"]
                    candidates.append(np.asarray(c, dtype=np.float32))
                candidates = np.stack(candidates)  # (K, 50, 7)
                obs_features = example["observation/state"]
                if modeb_active:
                    # Plan A: deadreckon mode-B scoring, no verifier needed
                    current_eef = np.asarray(obs_features[:3], dtype=np.float64)
                    scores = mode_b_score_batch(
                        candidates, current_eef,
                        mode_a_obj_pos, mode_b_obj_pos,
                        num_actions=modeb_num_actions,
                    )
                else:
                    scores = scorer_fn(obs_features, candidates, task_id=task_id)
                best_idx = int(np.argmax(scores))
                chunk = candidates[best_idx]
            chunk_counter += 1
            action_plan.extend(np.asarray(chunk[:replan_steps], dtype=np.float32))

        action = action_plan.popleft()
        obs, _, done, _ = env.step(action.tolist())
        t += 1

        # HDCSPI: check if current subtask's target object has been moved enough
        # to count as "placed in basket" — if yes, switch to next subtask's prompt.
        if hdcspi_subtasks and hdcspi_idx < len(hdcspi_subtasks) - 1:
            target_name = hdcspi_subtasks[hdcspi_idx]["target_object"] + "_pos"
            thresh = hdcspi_subtasks[hdcspi_idx]["displacement_threshold"]
            if target_name in initial_object_positions and target_name in obs:
                init_pos = np.asarray(initial_object_positions[target_name][:3], dtype=np.float64)
                cur_pos = np.asarray(obs[target_name][:3], dtype=np.float64)
                displacement = float(np.linalg.norm(cur_pos - init_pos))
                z_drop = float(init_pos[2] - cur_pos[2])
                # Completion: object substantially displaced AND has dropped (placed)
                if displacement > thresh and z_drop > 0.03:
                    hdcspi_log.append({
                        "t": t, "from_subtask": hdcspi_idx,
                        "completed_target": hdcspi_subtasks[hdcspi_idx]["target_object"],
                        "displacement": displacement, "z_drop": z_drop,
                    })
                    hdcspi_idx += 1
                    effective_language = hdcspi_subtasks[hdcspi_idx]["prompt"]
                    action_plan.clear()  # force re-plan with new prompt
                    print(f"  [HDCSPI t={t}] subtask {hdcspi_idx - 1} done "
                          f"(disp={displacement:.3f}m, dz={z_drop:.3f}m); "
                          f"switching to subtask {hdcspi_idx}: '{effective_language}'", flush=True)

        if t % WAYPOINT_EVERY == 0:
            waypoints.append({
                "t": t,
                "eef_pos": [float(x) for x in obs["robot0_eef_pos"]],
                "object_positions": _abs_obj_pos(obs),
            })
        if frames is not None and t % frame_stride == 0:
            try:
                if video_mode == "agentview":
                    img = np.asarray(obs["agentview_image"])[::-1]
                elif video_mode == "wrist":
                    img = np.asarray(obs["robot0_eye_in_hand_image"])[::-1]
                elif video_mode == "both":
                    a = np.asarray(obs["agentview_image"])[::-1]
                    w = np.asarray(obs["robot0_eye_in_hand_image"])[::-1]
                    img = np.concatenate([a, w], axis=1)
                else:
                    img = None
                if img is not None:
                    frames.append(img.astype(np.uint8))
            except KeyError:
                pass

    term = {
        "eef_pos": [float(x) for x in obs["robot0_eef_pos"]],
        "eef_quat": [float(x) for x in obs["robot0_eef_quat"]],
        "gripper_qpos": [float(x) for x in obs["robot0_gripper_qpos"]],
    }
    final_object_positions = _abs_obj_pos(obs)

    # Derived moved_objects list (parity with mode_collapse_check.py output)
    OBJ_MOVED_THRESHOLD_M = 0.02
    moved_objects = []
    for name, init_pos in initial_object_positions.items():
        if name in final_object_positions:
            delta = float(np.linalg.norm(
                np.array(init_pos) - np.array(final_object_positions[name])
            ))
            if delta > OBJ_MOVED_THRESHOLD_M:
                moved_objects.append({
                    "name": name,
                    "displacement_m": delta,
                    "initial": init_pos,
                    "final": final_object_positions[name],
                })
    moved_objects.sort(key=lambda x: -x["displacement_m"])

    env.close()
    return (bool(done), term, int(t), initial_object_positions,
            final_object_positions, waypoints, moved_objects, frames)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--n-tasks", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--eta", type=float, default=0.2,
                        help="SDE eta for conditions 2, 4, 6, 7")
    parser.add_argument("--replan-steps", type=int, default=REPLAN_STEPS,
                        help="Number of action steps to execute per replan (default 5)")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Suffix appended to OUT_DIR for this run (e.g., 'N150', 'replan25')")
    parser.add_argument("--task-ids", type=int, nargs="+", default=None,
                        help="Specific task IDs to run (e.g., '5 6 7 8 9'). Overrides --n-tasks.")
    parser.add_argument("--video-mode", choices=["off", "agentview", "wrist", "both"],
                        default="off",
                        help="If not 'off', save MP4 videos per trial (up to --video-max-per-cond per condition).")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--frame-stride", type=int, default=2,
                        help="Keep every Nth env step in video (1 = all, 2 = half).")
    parser.add_argument("--video-max-per-cond", type=int, default=10,
                        help="Cap on videos saved per TASK per condition to control disk usage. "
                             "E.g. cap=20 with 2 tasks saves up to 40 videos per condition.")
    parser.add_argument("--bddl-override", type=str, default=None,
                        help='JSON mapping task_id -> custom BDDL path, e.g. '
                             '\'{"0": "/path/T0_custom.bddl"}\'. Overrides LIBERO defaults. '
                             'Init states still come from the original LIBERO task '
                             '(scene topology must match).')
    parser.add_argument("--bddl-sweep-list", type=str, default=None,
                        help='JSON list of BDDL paths for SWEEP mode. Model is loaded ONCE '
                             'and each BDDL is run as a separate variant against task_id 0 '
                             '(or task_ids if specified). Output goes to '
                             'multimode_matrix_<out_suffix>_<variant_stem>/ per variant. '
                             'Mutually exclusive with --bddl-override.')
    parser.add_argument("--hdcspi", type=str, default=None,
                        help='JSON for HDCSPI sequential subtask execution. Format: '
                             '\'[{"prompt": "pick up X and put it in basket", '
                             '"target_object": "X_1", "displacement_threshold": 0.15}, ...]\'. '
                             'Pi0.5 starts with subtask[0].prompt. When subtask[i].target_object '
                             'has moved > threshold from initial position, switches to '
                             'subtask[i+1].prompt. Demonstrates mode-controlled execution by '
                             'sequencing trained single-target prompts.')
    parser.add_argument("--task-suite", type=str, default="libero_10",
                        choices=["libero_10", "libero_90", "libero_spatial",
                                 "libero_object", "libero_goal", "libero_long"],
                        help="LIBERO task suite to use. Default libero_10. "
                             "Use libero_90 for short-horizon single-target tasks "
                             "(needed for multi-instance scenes like KITCHEN_SCENE2).")
    parser.add_argument("--eta-schedule", type=str, default=None,
                        help='JSON for hybrid η schedule, e.g. '
                             '\'{"high": 0.8, "low": 0.0, "threshold": 0.5}\'. '
                             'When set, overrides the per-condition eta — η_high applies '
                             'to early flow integration (time > threshold) and η_low to '
                             'late integration. Targets the commitment window for '
                             'mode flipping while preserving execution precision.')
    parser.add_argument("--modeb-bon",
                        type=str, default=None,
                        help='JSON config for Plan A (mode-B exemplar BoN). Format: '
                             '\'{"k": 8, "deadreckon_n": 20}\'. When set, candidate chunks '
                             'are scored by deadreckoning their action sequences and '
                             'selecting the one whose direction best aligns with the '
                             'task-specific mode-B object (hardcoded per task_id).')
    parser.add_argument("--policy-config", type=str, default="pi0_libero",
                        help="openpi config name. Examples: pi0_libero (default), "
                             "pi05_libero (the pi0.5 LIBERO fine-tuned variant), "
                             "pi0_fast_libero. The config determines model architecture + base.")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Optional local checkpoint directory. If omitted, openpi will "
                             "auto-fetch from gs://openpi-assets/checkpoints/<config>/ on "
                             "first inference. Default for pi0_libero is the existing local "
                             "~/flare/checkpoints/pi0_libero_pt path.")
    parser.add_argument("--directed-noise", type=str, default=None,
                        help='JSON for directed-noise hybrid scheduler (Track 2). Format: '
                             '\'{"bias_strength": 0.5, "high": 0.8, "low": 0.0, "threshold": 0.5, '
                             '"target": "mode_b"}\'. When set, adds a per-task directional '
                             'bias to the noise during the high-noise window. Breaks Tweedie '
                             'invariance to enable mode-directed exploration.')
    args = parser.parse_args()

    # Parse bddl-override JSON once
    bddl_overrides: dict[int, str] = {}
    if args.bddl_override:
        try:
            raw = json.loads(args.bddl_override)
            bddl_overrides = {int(k): str(v) for k, v in raw.items()}
        except (json.JSONDecodeError, ValueError) as e:
            parser.error(f"--bddl-override JSON parse error: {e}")
        print(f"[bddl-override] active for tasks: {sorted(bddl_overrides.keys())}")
        for tid, p in sorted(bddl_overrides.items()):
            if not Path(p).exists():
                parser.error(f"--bddl-override path missing for task {tid}: {p}")
            print(f"  T{tid} -> {p}")

    # Parse SWEEP mode (Plan: 20-prompt sweep on T0). Mutually exclusive with --bddl-override.
    sweep_variants: list[tuple[str, dict[int, str]]] = []  # list of (variant_name, bddl_overrides)
    if args.bddl_sweep_list:
        if args.bddl_override:
            parser.error("--bddl-sweep-list and --bddl-override are mutually exclusive.")
        try:
            sweep_paths = json.loads(args.bddl_sweep_list)
            if not isinstance(sweep_paths, list):
                raise ValueError("--bddl-sweep-list must be a JSON list of paths")
        except (json.JSONDecodeError, ValueError) as e:
            parser.error(f"--bddl-sweep-list parse error: {e}")
        # Use the first task_id from --task-ids (default 0). This means the BDDL
        # override applies to that task. For multi-task sweeps, the override
        # applies to all listed tasks.
        sweep_target_tasks = list(args.task_ids) if args.task_ids else [0]
        for p in sweep_paths:
            if not Path(p).exists():
                parser.error(f"--bddl-sweep-list path missing: {p}")
            variant_name = Path(p).stem  # e.g. "T0_A1_reorder_both"
            # Map each target task to this variant's BDDL
            override_map = {tid: p for tid in sweep_target_tasks}
            sweep_variants.append((variant_name, override_map))
        print(f"[sweep] Running {len(sweep_variants)} BDDL variants on task_ids "
              f"{sweep_target_tasks} (model loaded ONCE for all)")
        for vn, vo in sweep_variants:
            print(f"  {vn}: {vo[0]}")

    # imageio availability check
    if args.video_mode != "off" and not HAS_IMAGEIO:
        print("[!] imageio not available — disabling video recording.")
        print("    Install:  uv pip install imageio imageio-ffmpeg")
        args.video_mode = "off"

    # Allow per-run output suffix so we don't clobber prior results
    global OUT_DIR
    if args.out_suffix:
        OUT_DIR = Path.home() / f"flare/results/multimode_matrix_{args.out_suffix}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Apply alpha + eta to conditions
    for cid in CONDITIONS:
        CONDITIONS[cid]["alpha"] = args.alpha
        if CONDITIONS[cid]["eta"] > 0:
            CONDITIONS[cid]["eta"] = args.eta

    selected = [CONDITIONS[c] for c in args.conditions]
    print("Conditions to run:")
    for c in selected:
        print(f"  {c['name']}: η={c['eta']}, K={c['K']}, "
              f"guidance={c['guidance']}, α={c.get('alpha', 0)}")

    print(f"\nLoading policy (post-patch) — config={args.policy_config}...", flush=True)
    Pi0 = apply_patch()
    cfg = _c.get_config(args.policy_config)
    # Pick checkpoint dir:
    # - if --checkpoint-dir set, use it
    # - else if pi0_libero (default), use the existing local CHECKPOINT_DIR
    # - else try to auto-download from gs://openpi-assets/checkpoints/<config>/params
    if args.checkpoint_dir is not None:
        ckpt_dir = args.checkpoint_dir
    elif args.policy_config == "pi0_libero":
        ckpt_dir = CHECKPOINT_DIR
    else:
        # Download the FULL checkpoint directory (includes both params/ AND assets/).
        # openpi's create_trained_policy expects:
        #   ckpt_dir/params/_METADATA   (model weights)
        #   ckpt_dir/assets/<asset_id>/norm_stats.json   (normalization)
        gs_path = f"gs://openpi-assets/checkpoints/{args.policy_config}"
        print(f"  [policy] no local checkpoint dir; attempting auto-download from {gs_path}",
              flush=True)
        try:
            from openpi.shared import download as openpi_download
            ckpt_dir = str(Path(openpi_download.maybe_download(gs_path)))
            print(f"  [policy] auto-downloaded full checkpoint to {ckpt_dir}", flush=True)
        except Exception as e:
            parser.error(
                f"Could not auto-download checkpoint for {args.policy_config}: {e}\n"
                f"Please pre-download manually:\n"
                f"  gsutil -m cp -r {gs_path} ~/.cache/openpi/openpi-assets/checkpoints/\n"
                f"then re-run with "
                f"--checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/{args.policy_config}"
            )
    policy = policy_config.create_trained_policy(cfg, ckpt_dir)
    model = None
    for attr in ["_model", "model", "_policy", "policy"]:
        if hasattr(policy, attr) and isinstance(getattr(policy, attr), Pi0):
            model = getattr(policy, attr)
            print(f"  found model under policy.{attr}", flush=True)
            break
    if model is None:
        # pi0.5 (and possibly other configs) may not expose a PI0Pytorch instance —
        # it might be a JAX-backed model. The SDE/guidance patches won't apply,
        # but pure ODE inference (eta=0, no guidance, no BoN scorer requiring obs_state)
        # works fine via policy.infer().
        print("  [WARN] no PI0Pytorch instance found on policy; using dummy model.", flush=True)
        print("  [WARN] SDE/guidance/directed-noise will be silently no-op.", flush=True)
        print("  [WARN] This is OK for ODE-only experiments. For SDE/guided, results will be invalid.",
              flush=True)
        import types
        model = types.SimpleNamespace()

    # Initialize model attributes used by patch (defaults)
    model.flare_eta = 0.0
    model.flare_verifier_fn = None
    model.flare_alpha = 0.0
    model.flare_obs_state = None
    model.flare_eta_high = None
    model.flare_eta_low = None
    model.flare_eta_threshold = 0.5

    # Apply optional hybrid η schedule (Plan B)
    if args.eta_schedule:
        try:
            sched = json.loads(args.eta_schedule)
            model.flare_eta_high = float(sched["high"])
            model.flare_eta_low = float(sched["low"])
            model.flare_eta_threshold = float(sched.get("threshold", 0.5))
            print(f"[eta-schedule] ACTIVE: high={model.flare_eta_high}, "
                  f"low={model.flare_eta_low}, threshold={model.flare_eta_threshold} "
                  f"(η_high applies when integration time > threshold)", flush=True)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            parser.error(f"--eta-schedule parse error: {e}")

    # Parse Plan A config (mode-B deadreckoning BoN)
    modeb_bon_config = None
    if args.modeb_bon:
        try:
            modeb_bon_config = json.loads(args.modeb_bon)
        except json.JSONDecodeError as e:
            parser.error(f"--modeb-bon parse error: {e}")
        print(f"[modeb-bon] ACTIVE: config={modeb_bon_config}", flush=True)
        print(f"  Will score candidates by deadreckoning the first "
              f"{modeb_bon_config.get('deadreckon_n', 20)} delta-eef actions", flush=True)

    # Parse HDCSPI subtask list (Hierarchical Decomposition with Cross-Suite Prompt Injection)
    hdcspi_subtasks = None
    if args.hdcspi:
        try:
            hdcspi_subtasks = json.loads(args.hdcspi)
            if not isinstance(hdcspi_subtasks, list) or not hdcspi_subtasks:
                raise ValueError("--hdcspi must be a non-empty JSON list")
            for st in hdcspi_subtasks:
                if "prompt" not in st or "target_object" not in st:
                    raise ValueError("each subtask needs 'prompt' and 'target_object' keys")
                # Default displacement threshold for completion detection
                st.setdefault("displacement_threshold", 0.15)
        except (json.JSONDecodeError, ValueError) as e:
            parser.error(f"--hdcspi parse error: {e}")
        print(f"[HDCSPI] {len(hdcspi_subtasks)} subtasks configured:", flush=True)
        for i, st in enumerate(hdcspi_subtasks):
            print(f"  {i}: prompt='{st['prompt']}' target='{st['target_object']}' "
                  f"thresh={st['displacement_threshold']}", flush=True)

    # Parse directed-noise config (Track 2)
    directed_noise_config = None
    if args.directed_noise:
        try:
            directed_noise_config = json.loads(args.directed_noise)
        except json.JSONDecodeError as e:
            parser.error(f"--directed-noise parse error: {e}")
        # Apply eta_high/low/threshold from this config to the model attributes
        # (overrides any --eta-schedule, since directed noise needs a schedule window)
        model.flare_eta_high = float(directed_noise_config["high"])
        model.flare_eta_low = float(directed_noise_config["low"])
        model.flare_eta_threshold = float(directed_noise_config.get("threshold", 0.5))
        # bias_strength sets here; bias_direction is set per-task in rollout
        model.flare_noise_bias_strength = float(directed_noise_config["bias_strength"])
        print(f"[directed-noise] ACTIVE: high={model.flare_eta_high}, "
              f"low={model.flare_eta_low}, threshold={model.flare_eta_threshold}, "
              f"bias_strength={model.flare_noise_bias_strength}, "
              f"target={directed_noise_config.get('target', 'mode_b')}",
              flush=True)
        print(f"  Bias direction will be computed per-task from object positions "
              f"(target={directed_noise_config.get('target', 'mode_b')})", flush=True)

    print("\nBuilding verifier callables...", flush=True)
    scorers = build_scorers(v3mc_ckpt=V3MC_CKPT, wmt_ckpt=WMT_CKPT,
                            v2_cache=V2_CACHE, device="cuda")
    print(f"  scorers available: {list(scorers.keys())}", flush=True)
    guidance_callables = {
        "V3_MC": V3MCGuidanceCallable(V3MC_CKPT, device="cuda"),
        "WM_terminal": WMTGuidanceCallable(WMT_CKPT, device="cuda"),
    }
    print(f"  guidance callables: {list(guidance_callables.keys())}", flush=True)

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    rollout._task_suite = args.task_suite  # propagate to rollout for max_steps lookup
    print(f"[task-suite] using {args.task_suite} ({suite.n_tasks} tasks)", flush=True)
    if args.task_ids is not None:
        task_id_list = list(args.task_ids)
        print(f"  Running specific tasks: {task_id_list}", flush=True)
    else:
        n_tasks = min(args.n_tasks, suite.n_tasks)
        task_id_list = list(range(n_tasks))

    # Build sweep iteration list. If no sweep_list, use single-iteration default.
    if not sweep_variants:
        sweep_variants = [(None, bddl_overrides)]
    base_suffix = args.out_suffix
    base_out_dir = OUT_DIR
    total_t_start = time.time()

    for variant_idx, (variant_name, variant_overrides) in enumerate(sweep_variants):
        if variant_name is not None:
            sweep_suffix = f"{base_suffix}_{variant_name}" if base_suffix else variant_name
            OUT_DIR = Path.home() / f"flare/results/multimode_matrix_{sweep_suffix}"
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            print(f"\n{'═' * 72}", flush=True)
            print(f"═══ SWEEP VARIANT [{variant_idx + 1}/{len(sweep_variants)}]: {variant_name}",
                  flush=True)
            print(f"═══ OUT_DIR: {OUT_DIR}", flush=True)
            print(f"{'═' * 72}", flush=True)
        else:
            OUT_DIR = base_out_dir
        bddl_overrides = variant_overrides

        all_results = {}
        t_global = time.time()
        for cond in selected:
            cond_name = cond["name"]
            print(f"\n=== Condition: {cond_name} ===", flush=True)
            t_cond = time.time()
            results = []
            scorer_fn = scorers.get(cond["verifier"]) if cond["verifier"] else None
            guide_verif_name = cond.get("guide_verif", None)
            guidance_obj = guidance_callables.get(guide_verif_name) if guide_verif_name else None
    
            # Per-condition video output dir (only created if video_mode != off)
            cond_video_dir = None
            if args.video_mode != "off":
                cond_video_dir = OUT_DIR / f"videos_{cond_name}"
                cond_video_dir.mkdir(parents=True, exist_ok=True)
                print(f"  [video] mode={args.video_mode}, dir={cond_video_dir}, "
                      f"cap={args.video_max_per_cond}/task/cond", flush=True)
            # Per-task counter so we save up to N videos for EACH task, not N total per cond
            videos_saved_per_task = collections.defaultdict(int)
    
            for task_id in task_id_list:
                task = suite.get_task(task_id)
                if task_id in bddl_overrides:
                    bddl = bddl_overrides[task_id]
                    language_override = read_bddl_language(bddl)
                    print(f"  [T{task_id}] using override BDDL: {bddl}", flush=True)
                    if language_override is not None:
                        print(f"  [T{task_id}] override language: {language_override!r}", flush=True)
                    else:
                        print(f"  [T{task_id}] WARNING: override BDDL had no parseable (:language ...) section; "
                              f"falling back to LIBERO task.language={task.language!r}", flush=True)
                else:
                    bddl = os.path.join(get_libero_path("bddl_files"),
                                        task.problem_folder, task.bddl_file)
                    language_override = None
                init_states = suite.get_task_init_states(task_id)
                for trial in range(args.n_trials):
                    init_idx = trial % len(init_states)
                    env_seed = 42 + trial
                    base_seed = (BASE_SEED + 1000 * int(cond["eta"] * 10) +
                                 73 * trial + 31 * task_id +
                                 1009 * (1 if cond["guidance"] else 0) +
                                 71 * cond["K"])
                    t0 = time.time()
                    # Pass video flags only if per-task cap not yet hit for this condition
                    this_video_mode = args.video_mode if (
                        args.video_mode != "off"
                        and videos_saved_per_task[task_id] < args.video_max_per_cond
                    ) else "off"
                    (succ, term, n_steps, init_obj_pos, final_obj_pos,
                     wps, moved_objs, frames) = rollout(
                        policy, model, task, task_id, bddl, init_states[init_idx],
                        env_seed, base_seed, cond, scorer_fn, guidance_obj,
                        obs_state_for_guidance=None,  # set per-call inside rollout
                        replan_steps=args.replan_steps,
                        video_mode=this_video_mode, frame_stride=args.frame_stride,
                        language_override=language_override,
                        modeb_bon_config=modeb_bon_config,
                        hdcspi_subtasks=hdcspi_subtasks,
                    )
                    elapsed = time.time() - t0
                    # Save video if collected
                    if frames and cond_video_dir is not None:
                        vname = (f"t{task_id}_trial{trial:02d}_init{init_idx:02d}_"
                                 f"succ{succ}_steps{n_steps}.mp4")
                        vpath = cond_video_dir / vname
                        try:
                            imageio.mimsave(str(vpath), frames,
                                            fps=args.video_fps, macro_block_size=1)
                            videos_saved_per_task[task_id] += 1
                            print(f"  [video] saved {vpath.name} "
                                  f"({len(frames)} frames)", flush=True)
                        except Exception as e:
                            print(f"  [video] save failed: {e}", flush=True)
                    results.append({
                        "task_id": task_id, "task": str(task.language),
                        "language_used": language_override if language_override is not None else str(task.language),
                        "bddl_path": bddl,
                        "trial": trial, "init_idx": init_idx,
                        "env_seed": env_seed,
                        "base_sampling_seed": base_seed,
                        "condition": cond_name, "eta": cond["eta"], "K": cond["K"],
                        "guidance": cond["guidance"], "alpha": cond.get("alpha", 0.0),
                        "success": succ, "steps": n_steps,
                        "wallclock_sec": float(elapsed), "terminal_state": term,
                        "initial_object_positions": init_obj_pos,
                        "final_object_positions": final_obj_pos,
                        "moved_objects": moved_objs,
                        "waypoints": wps,
                    })
                    n_done = len(results)
                    n_succ = sum(r["success"] for r in results)
                    # Rich per-trial log (matches mode_collapse_check.py style + condition tag)
                    moved_names = [m["name"] for m in moved_objs[:3]]
                    eef = term["eef_pos"]
                    print(f"  [{n_done:3d}/{len(task_id_list)*args.n_trials}] "
                          f"{cond_name} T{task_id} trial={trial:2d} init={init_idx:2d}: "
                          f"succ={succ} steps={n_steps} "
                          f"eef=({eef[0]:+.3f},{eef[1]:+.3f},{eef[2]:+.3f}) "
                          f"moved={moved_names} "
                          f"t={elapsed:.0f}s | "
                          f"running {n_succ}/{n_done} ({100*n_succ/n_done:.0f}%)",
                          flush=True)
            cond_succ = sum(r["success"] for r in results)
            cond_rate = cond_succ / max(len(results), 1)
            print(f"  >> {cond_name}: {cond_succ}/{len(results)} = {cond_rate:.0%}  "
                  f"in {time.time() - t_cond:.0f}s", flush=True)
            all_results[cond_name] = results
            with open(OUT_DIR / f"{cond_name}.json", "w") as f:
                json.dump(results, f, indent=2)
    
        # ---- Summary ----
        print("\n==== SUMMARY ====", flush=True)
        summary = {"args": vars(args), "per_cond": {}}
        for cond_name, results in all_results.items():
            rate = sum(r["success"] for r in results) / max(len(results), 1)
            wall_total = sum(r["wallclock_sec"] for r in results)
            summary["per_cond"][cond_name] = {
                "n_episodes": len(results),
                "n_successes": int(sum(r["success"] for r in results)),
                "success_rate": float(rate),
                "wallclock_total_sec": float(wall_total),
                "wallclock_per_ep_sec": float(wall_total / max(len(results), 1)),
            }
            print(f"  {cond_name}: {summary['per_cond'][cond_name]}", flush=True)
        summary["total_wallclock_hr"] = float((time.time() - t_global) / 3600)
        summary["replan_steps"] = args.replan_steps
        with open(OUT_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    # Reset model
    model.flare_eta = 0.0
    model.flare_verifier_fn = None
    model.flare_alpha = 0.0
    print(f"\nDone in {(time.time() - t_global)/3600:.1f}h. Saved to {OUT_DIR}/.",
          flush=True)


if __name__ == "__main__":
    main()

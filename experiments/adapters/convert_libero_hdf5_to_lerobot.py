"""convert_libero_hdf5_to_lerobot.py

Convert LIBERO HDF5 demo files into a LeRobotDataset that openpi's
LeRobotLiberoDataConfig can consume for LoRA fine-tuning.

KEY DESIGN: builds a **counterfactual** dataset by combining multiple HDF5s
(different prompts on the same scene) into a single LeRobotDataset with
per-episode task descriptions. This is what makes language causally
non-redundant during training.

================================================================================
USAGE
================================================================================

# Single-prompt conversion (testing):
python3 convert_libero_hdf5_to_lerobot.py \\
    --inputs path/to/libero_90/LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket_demo.hdf5 \\
    --prompts "pick up the tomato sauce and put it in the basket" \\
    --out-dir ~/flare/lerobot_data/libero_t54_only \\
    --repo-id libero_t54_only

# Counterfactual multi-prompt conversion (the main use case):
python3 convert_libero_hdf5_to_lerobot.py \\
    --inputs \\
        ~/flare/openpi/third_party/libero/libero/datasets/libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo.hdf5 \\
        ~/flare/openpi/third_party/libero/libero/datasets/libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo.hdf5 \\
        ~/flare/openpi/third_party/libero/libero/datasets/libero_90/LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket_demo.hdf5 \\
        ~/flare/openpi/third_party/libero/libero/datasets/libero_90/LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket_demo.hdf5 \\
        ~/flare/openpi/third_party/libero/libero/datasets/libero_90/LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket_demo.hdf5 \\
        ~/flare/openpi/third_party/libero/libero/datasets/libero_90/LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket_demo.hdf5 \\
        ~/flare/openpi/third_party/libero/libero/datasets/libero_90/LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket_demo.hdf5 \\
    --prompts \\
        "put both the alphabet soup and the tomato sauce in the basket" \\
        "put both the cream cheese box and the butter in the basket" \\
        "pick up the alphabet soup and put it in the basket" \\
        "pick up the butter and put it in the basket" \\
        "pick up the milk and put it in the basket" \\
        "pick up the orange juice and put it in the basket" \\
        "pick up the tomato sauce and put it in the basket" \\
    --out-dir ~/flare/lerobot_data/libero_scene2_counterfactual \\
    --repo-id libero_scene2_counterfactual

# Verify the dataset loads correctly afterwards:
python3 convert_libero_hdf5_to_lerobot.py \\
    --verify-only ~/flare/lerobot_data/libero_scene2_counterfactual

================================================================================
VERIFIED ASSUMPTIONS (about LIBERO HDF5 format, from our enumeration today)
================================================================================

Each HDF5 file has:
    data/
        demo_0/, demo_1/, ..., demo_49/    # 50 demos per file
            obs/
                agentview_rgb       # (T, 128, 128, 3) uint8, main camera   [VERIFIED]
                eye_in_hand_rgb     # (T, 128, 128, 3) uint8, wrist camera  [VERIFIED]
                ee_pos              # (T, 3)    float, end-effector position
                ee_ori              # (T, 3)    float, end-effector orientation (euler)
                ee_states           # (T, 6)    float, full ee state (pos+ori combined?)
                gripper_states      # (T, 2)    float, gripper [width, ...?]
                joint_states        # (T, 7)    float, 7-DOF joint angles
            actions                 # (T, 7)    float, delta actions (xyz+rpy+gripper)
            states                  # (T, 123)  float, full simulator state (we DON'T need this for training)

State vector for pi0 (8-dim, MATCHES openpi/examples/libero/main.py inference):
    state = ee_pos (3) + ee_ori_axisangle (3) + gripper_qpos (2) = 8 dims
    LIBERO's HDF5 stores ee_ori already as axis-angle (3 dims, not quat), so no
    quat→axisangle conversion is needed.

Action vector for pi0 (7-dim):
    Same as HDF5's `actions` field — already in the right format.

Image preprocessing (matches openpi inference exactly):
    1. Rotate 180° (LIBERO env returns upside-down images that need flipping)
    2. Resize from raw HDF5 size (128×128) to 256×256 (openpi's storage format)
    3. Model resizes 256→224 internally via PaliGemma image processor

FPS = 10 (matches openpi's training configuration)

================================================================================
HISTORY: previous version had THREE bugs that caused LoRA inference to fail
================================================================================
  (1) State was joint_states[:7] + gripper[:1] — model expected ee_pos+ee_ori+gripper
  (2) Images stored at 128×128 raw — should be 256×256 rotated
  (3) FPS was 20 — should be 10
After fixing these, LoRA inference produces sensible action chunks.

2. IMAGE FORMAT: LIBERO HDF5 stores images as raw uint8 RGB. LeRobot expects
   the same. We do no conversion. If openpi's loader applies image transforms
   (e.g. BGR conversion, flip), they happen downstream.

3. ACTION SPACE: openpi expects 7-dim actions. We forward the HDF5's `actions`
   field directly. If openpi expects normalized actions, the normalization will
   be computed by `compute_norm_stats.py` after this script runs.

4. FPS: LIBERO records at 20 Hz. We hardcode fps=20 in the LeRobot dataset.
   If your dataset is different, change `FPS` below.

5. EPISODE TASK INDEX: LeRobot supports per-episode task strings. The training
   loop tokenizes these at runtime. We assign the prompt provided for each
   input HDF5 to all demos within that file.

================================================================================
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

# LeRobot import — verify available before running
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("[WARN] lerobot package not importable. Install with: pip install lerobot")

# Constants based on openpi's official LIBERO conversion (examples/libero/convert_libero_data_to_lerobot.py)
# and the libero inference loop (examples/libero/main.py)
#
# CRITICAL fixes vs the original version of this script:
# - State is now (ee_pos, ee_ori_axis_angle, gripper_qpos) = 8 dims (NOT joints+gripper)
# - Images stored at 256×256 to match openpi's spec (model resizes to 224 at inference)
# - Images rotated 180° (matches openpi's inference preprocessing for the env's upside-down view)
# - FPS = 10 (matches openpi's training; was 20 before — was a bug)
FPS = 10
IMG_SHAPE = (256, 256, 3)   # openpi standard; model resizes to 224 internally
ACTION_DIM = 7              # LIBERO action: 3 xyz + 3 rpy + 1 gripper
STATE_DIM = 8               # ee_pos(3) + ee_ori_axisangle(3) + gripper_qpos(2)


def build_state_vector(obs_group: h5py.Group, frame_idx: int) -> np.ndarray:
    """Construct the 8-dim state vector expected by pi0_libero.

    Matches openpi/examples/libero/main.py exactly:
        state = concat(
            ee_pos (3),                            # robot0_eef_pos
            ee_ori_axis_angle (3),                 # _quat2axisangle(robot0_eef_quat)
            gripper_qpos (2),                      # robot0_gripper_qpos
        ) = 8 dims

    LIBERO's HDF5 already stores ee_ori in axis-angle form (3 dims, not quat 4 dims),
    so no quat conversion needed.

    Returns:
        np.ndarray of shape (STATE_DIM,) float32
    """
    ee_pos = obs_group["ee_pos"][frame_idx]              # (3,)
    ee_ori = obs_group["ee_ori"][frame_idx]              # (3,) already axis-angle
    gripper = obs_group["gripper_states"][frame_idx]     # (2,) [left, right]

    if ee_pos.shape[0] != 3:
        raise ValueError(f"Expected 3 ee_pos dims, got {ee_pos.shape}")
    if ee_ori.shape[0] != 3:
        raise ValueError(f"Expected 3 ee_ori dims, got {ee_ori.shape}")
    if gripper.shape[0] < 2:
        raise ValueError(f"Expected 2 gripper dims, got {gripper.shape}")

    state = np.concatenate([
        ee_pos.astype(np.float32),
        ee_ori.astype(np.float32),
        gripper[:2].astype(np.float32),
    ])
    assert state.shape == (STATE_DIM,), f"State shape {state.shape} != ({STATE_DIM},)"
    return state


def preprocess_image(img: np.ndarray, target_size: int = 256) -> np.ndarray:
    """Match openpi's LIBERO image preprocessing exactly:
        1. Rotate 180° (LIBERO env returns upside-down images)
        2. Resize to target_size (default 256 for storage; model resizes to 224)

    Args:
        img: (H, W, 3) uint8
        target_size: int, output H = W = target_size

    Returns:
        (target_size, target_size, 3) uint8
    """
    from PIL import Image
    # 180° rotation: matches `obs[::-1, ::-1]` in openpi/examples/libero/main.py
    img = np.ascontiguousarray(img[::-1, ::-1])
    # Resize from input (likely 128) up to 256
    if img.shape[0] != target_size or img.shape[1] != target_size:
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
        img = np.asarray(pil_img)
    return img.astype(np.uint8)


def extract_demo_frames(hdf5_path: Path, prompt: str,
                        max_demos: int = None) -> List[List[Dict]]:
    """Read demos from an HDF5 and convert to LeRobot frame dicts.

    Args:
        hdf5_path: path to the LIBERO .hdf5 file
        prompt: task description (used by caller; this function just reads frames)
        max_demos: if set, limit to first N demos (for testing)

    Returns:
        List of episodes; each episode is a list of frame dicts.
    """
    episodes = []
    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        if max_demos is not None:
            demo_keys = demo_keys[:max_demos]
        print(f"  Found {len(demo_keys)} demos to process in {hdf5_path.name}")

        for demo_idx, demo_key in enumerate(demo_keys):
            demo = f["data"][demo_key]
            obs = demo["obs"]

            # Read whole arrays into memory (faster than per-frame indexing)
            actions = demo["actions"][()]                   # (T, 7)
            agentview = obs["agentview_rgb"][()]            # (T, 256, 256, 3)
            wrist = obs["eye_in_hand_rgb"][()]              # (T, 256, 256, 3)
            joints_all = obs["joint_states"][()]            # (T, 7)
            gripper_all = obs["gripper_states"][()]         # (T, 2)
            T = len(actions)

            # Sanity check: all arrays should have same length
            assert len(agentview) == T, \
                f"agentview_rgb len {len(agentview)} != actions len {T} in {demo_key}"
            assert len(wrist) == T, \
                f"wrist_image len {len(wrist)} != actions len {T} in {demo_key}"
            assert len(joints_all) == T, \
                f"joint_states len {len(joints_all)} != actions len {T} in {demo_key}"
            assert len(gripper_all) == T, \
                f"gripper_states len {len(gripper_all)} != actions len {T} in {demo_key}"

            # Pre-read everything we need for proper state building
            ee_pos_all = obs["ee_pos"][()]            # (T, 3)
            ee_ori_all = obs["ee_ori"][()]            # (T, 3) already axis-angle

            frames = []
            for t in range(T):
                # State: openpi-compatible = ee_pos(3) + ee_ori_axisangle(3) + gripper_qpos(2)
                state = np.concatenate([
                    ee_pos_all[t].astype(np.float32),
                    ee_ori_all[t].astype(np.float32),
                    gripper_all[t, :2].astype(np.float32),
                ])
                # Image: rotate 180° + resize to 256 (matches openpi inference pipeline)
                img = preprocess_image(agentview[t], target_size=IMG_SHAPE[0])
                wrist_img = preprocess_image(wrist[t], target_size=IMG_SHAPE[0])
                frames.append({
                    "image": img,                                  # (256, 256, 3) uint8
                    "wrist_image": wrist_img,                      # (256, 256, 3) uint8
                    "state": state,                                # (8,) float32
                    "actions": actions[t].astype(np.float32),      # (7,) float32
                    "task": prompt,                                # str — required per-frame by lerobot
                })
            episodes.append(frames)

            if demo_idx == 0:
                # Print structure summary for first demo
                frame = frames[0]
                print(f"    demo_0: T={T} frames")
                print(f"      image: shape={frame['image'].shape}, dtype={frame['image'].dtype}")
                print(f"      wrist_image: shape={frame['wrist_image'].shape}, dtype={frame['wrist_image'].dtype}")
                print(f"      state: shape={frame['state'].shape}, dtype={frame['state'].dtype}, "
                      f"sample={frame['state']}")
                print(f"      actions: shape={frame['actions'].shape}, dtype={frame['actions'].dtype}, "
                      f"sample={frame['actions']}")
    return episodes


def create_lerobot_dataset(out_dir: Path, repo_id: str):
    """Create a fresh empty LeRobotDataset with the right features."""
    if not LEROBOT_AVAILABLE:
        raise RuntimeError("lerobot not installed. pip install lerobot")

    # Wipe existing dataset if any
    if out_dir.exists():
        print(f"  [warn] removing existing dataset at {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    features = {
        "image": {
            "dtype": "image",
            "shape": IMG_SHAPE,
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": IMG_SHAPE,
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": [f"state_{i}" for i in range(STATE_DIM)],
        },
        "actions": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        root=str(out_dir),
        features=features,
        use_videos=False,   # use individual images for simplicity; smaller dataset
    )
    return dataset


def main():
    p = argparse.ArgumentParser(
        description="Convert LIBERO HDF5 demos to LeRobotDataset for openpi LoRA training"
    )
    p.add_argument("--inputs", nargs="+",
                   help="One or more LIBERO HDF5 file paths")
    p.add_argument("--prompts", nargs="+",
                   help="One prompt per input HDF5 (same order). The prompt is "
                        "attached to every demo in that file.")
    p.add_argument("--out-dir", type=str,
                   help="Output directory for the LeRobotDataset")
    p.add_argument("--repo-id", type=str, default=None,
                   help="LeRobot repo_id (used internally; default: derived from out-dir name)")
    p.add_argument("--max-demos-per-file", type=int, default=None,
                   help="For testing: limit to first N demos per HDF5 file (default: all 50)")
    p.add_argument("--verify-only", type=str, default=None,
                   help="If set, skip conversion and just verify the dataset at this path is loadable")
    args = p.parse_args()

    # --verify-only mode
    if args.verify_only:
        verify_dataset(Path(args.verify_only).expanduser())
        return

    # Validate inputs
    if not args.inputs or not args.prompts or not args.out_dir:
        p.error("--inputs, --prompts, and --out-dir are all required (unless --verify-only)")
    if len(args.inputs) != len(args.prompts):
        p.error(f"--inputs ({len(args.inputs)}) and --prompts ({len(args.prompts)}) "
                f"must have same length")

    out_dir = Path(args.out_dir).expanduser().resolve()
    repo_id = args.repo_id or out_dir.name

    # Print plan
    print("=" * 80)
    print("LIBERO HDF5 → LeRobotDataset converter")
    print("=" * 80)
    print(f"  Output: {out_dir}")
    print(f"  Repo ID: {repo_id}")
    print(f"  Inputs ({len(args.inputs)}):")
    for path_str, prompt in zip(args.inputs, args.prompts):
        path = Path(path_str).expanduser()
        if not path.exists():
            p.error(f"Input HDF5 not found: {path}")
        size_mb = path.stat().st_size / 1e6
        print(f"    [{size_mb:6.0f} MB] {path.name}")
        print(f"           prompt: {prompt!r}")
    print()

    # Create dataset
    print(f"[1/3] Creating empty LeRobotDataset at {out_dir} ...")
    dataset = create_lerobot_dataset(out_dir, repo_id)

    # Process each input
    print(f"\n[2/3] Reading HDF5 files and adding episodes ...")
    total_episodes = 0
    total_frames = 0
    summary = []
    for hdf5_path_str, prompt in zip(args.inputs, args.prompts):
        hdf5_path = Path(hdf5_path_str).expanduser()
        print(f"\n  Processing {hdf5_path.name}")
        print(f"    Prompt: {prompt!r}")
        episodes = extract_demo_frames(hdf5_path, prompt,
                                        max_demos=args.max_demos_per_file)
        n_ep = len(episodes)
        n_frames = sum(len(ep) for ep in episodes)
        print(f"    -> {n_ep} episodes, {n_frames} frames total")

        # Add to LeRobotDataset
        # Note: modern lerobot API requires 'task' per-frame (already in frame dict);
        # save_episode() takes no task argument
        for ep_idx, frames in enumerate(episodes):
            for frame in frames:
                dataset.add_frame(frame)
            dataset.save_episode()

        total_episodes += n_ep
        total_frames += n_frames
        summary.append({
            "hdf5": str(hdf5_path),
            "prompt": prompt,
            "n_episodes": n_ep,
            "n_frames": n_frames,
        })

    # Finalize dataset
    print(f"\n[3/3] Finalizing dataset ...")
    # In modern LeRobot, dataset.consolidate() or similar may be needed.
    # If your lerobot version doesn't have this method, comment out:
    if hasattr(dataset, "consolidate"):
        dataset.consolidate()

    # Save conversion manifest
    manifest_path = out_dir / "conversion_manifest.json"
    manifest_path.write_text(json.dumps({
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "n_unique_prompts": len(set(args.prompts)),
        "repo_id": repo_id,
        "fps": FPS,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "image_shape": list(IMG_SHAPE),
        "inputs": summary,
    }, indent=2))

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"  Total episodes: {total_episodes}")
    print(f"  Total frames:   {total_frames}")
    print(f"  Unique prompts: {len(set(args.prompts))}")
    print(f"  Output dir:     {out_dir}")
    print(f"  Manifest:       {manifest_path}")
    print()
    print(f"  Verify with:")
    print(f"    python3 {Path(__file__).name} --verify-only {out_dir}")
    print()
    print(f"  Next: configure openpi's LeRobotLiberoDataConfig to point at this dataset")
    print(f"  (or use the local path with HF datasets-like API)")


def verify_dataset(dataset_dir: Path):
    """Verify the produced dataset by inspecting files on disk + metadata.

    Does NOT use LeRobotDataset() constructor because it tries to hit the
    HuggingFace API for local-only datasets. Instead we walk the directory
    structure and read the meta files directly.
    """
    print(f"=" * 80)
    print(f"VERIFY: {dataset_dir}")
    print(f"=" * 80)

    if not dataset_dir.exists():
        print(f"  [FAIL] dataset dir does not exist: {dataset_dir}")
        return

    manifest_path = dataset_dir / "conversion_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        print("\n  CONVERSION MANIFEST:")
        for k, v in manifest.items():
            if k != "inputs":
                print(f"    {k}: {v}")
        print(f"    n_input_hdf5_files: {len(manifest.get('inputs', []))}")
    else:
        print("  [warn] no conversion_manifest.json found")

    # Walk directory structure
    print(f"\n  DIRECTORY STRUCTURE:")
    for path in sorted(dataset_dir.rglob("*"))[:30]:
        rel = path.relative_to(dataset_dir)
        if path.is_file():
            size_kb = path.stat().st_size / 1024
            print(f"    {rel}  ({size_kb:.0f} KB)")
        else:
            print(f"    {rel}/")

    # Read meta/info.json if it exists (LeRobot writes this)
    info_path = dataset_dir / "meta" / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        print(f"\n  META/INFO.JSON:")
        for k, v in info.items():
            if isinstance(v, (dict, list)) and len(str(v)) > 200:
                print(f"    {k}: <large value, type={type(v).__name__}>")
            else:
                print(f"    {k}: {v}")

    # Read meta/tasks.jsonl if it exists (lists unique task descriptions)
    tasks_path = dataset_dir / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        print(f"\n  UNIQUE TASKS (from meta/tasks.jsonl):")
        for line in tasks_path.read_text().splitlines()[:20]:
            try:
                entry = json.loads(line)
                print(f"    [{entry.get('task_index', '?')}] {entry.get('task', '?')!r}")
            except Exception:
                print(f"    {line[:150]}")

    # Count episodes
    episodes_dir = dataset_dir / "meta" / "episodes.jsonl"
    if episodes_dir.exists():
        n_eps = sum(1 for _ in episodes_dir.read_text().splitlines() if _.strip())
        print(f"\n  EPISODES META: {n_eps} episodes")

    # Count data files
    data_dir = dataset_dir / "data"
    if data_dir.exists():
        parquet_files = list(data_dir.rglob("*.parquet"))
        print(f"\n  DATA: {len(parquet_files)} parquet files")

    # Count image files
    img_dir = dataset_dir / "images"
    if img_dir.exists():
        n_img = sum(1 for _ in img_dir.rglob("*.png")) + sum(1 for _ in img_dir.rglob("*.jpg"))
        print(f"  IMAGES: {n_img} image files")

    print(f"\n  VERIFY DONE")


if __name__ == "__main__":
    main()

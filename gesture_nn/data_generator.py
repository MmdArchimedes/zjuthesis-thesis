"""
Synthetic hand skeleton data generator simulating Rokid UXR output.

Generates realistic 26-joint hand skeletons for 6 gesture classes
with biomechanical variation, sensor noise, and temporal dynamics.

Joint set follows Rokid UXR hand tracking convention:
  Wrist (1), Palm (1), Thumb (4: CMC/MCP/IP/Tip),
  Index (4: MCP/PIP/DIP/Tip), Middle (4), Ring (4), Pinky (4),
  plus additional intermediate joints to reach 26.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import pickle
from pathlib import Path
from config import *


@dataclass
class HandSkeleton:
    """Canonical hand skeleton in rest pose (open palm facing forward)."""
    joints: np.ndarray  # [26, 3] in meters, wrist-relative

    @classmethod
    def build_right_hand(cls) -> "HandSkeleton":
        """Build canonical right-hand skeleton with anatomically plausible proportions."""
        joints = np.zeros((26, 3), dtype=np.float32)

        # Wrist at origin
        wrist = 0
        joints[0] = [0.0, 0.0, 0.0]

        # Palm center
        joints[1] = [0.0, 0.02, 0.0]

        # Thumb: CMC → MCP → IP → Tip (joints 2-5)
        joints[2] = [0.015, 0.010, -0.005]   # CMC
        joints[3] = [0.025, 0.025, -0.010]   # MCP
        joints[4] = [0.030, 0.045, -0.012]   # IP
        joints[5] = [0.032, 0.060, -0.012]   # Tip

        # Index finger: MCP → PIP → DIP → Tip (joints 6-9)
        joints[6] = [0.020, 0.015, 0.005]    # MCP
        joints[7] = [0.018, 0.040, 0.005]    # PIP
        joints[8] = [0.016, 0.055, 0.005]    # DIP
        joints[9] = [0.015, 0.070, 0.005]    # Tip

        # Middle finger: MCP → PIP → DIP → Tip (joints 10-13)
        joints[10] = [0.0, 0.015, 0.008]      # MCP
        joints[11] = [0.0, 0.045, 0.008]      # PIP
        joints[12] = [0.0, 0.062, 0.008]      # DIP
        joints[13] = [0.0, 0.078, 0.008]      # Tip

        # Ring finger: MCP → PIP → DIP → Tip (joints 14-17)
        joints[14] = [-0.018, 0.015, 0.005]   # MCP
        joints[15] = [-0.016, 0.042, 0.005]   # PIP
        joints[16] = [-0.015, 0.058, 0.005]   # DIP
        joints[17] = [-0.014, 0.072, 0.005]   # Tip

        # Pinky: MCP → PIP → DIP → Tip (joints 18-21)
        joints[18] = [-0.028, 0.015, 0.0]     # MCP
        joints[19] = [-0.025, 0.035, 0.0]     # PIP
        joints[20] = [-0.024, 0.048, 0.0]     # DIP
        joints[21] = [-0.023, 0.058, 0.0]     # Tip

        # Additional tracking points (22-25): mid-phalanx estimates
        joints[22] = [0.017, 0.028, 0.005]    # index mid
        joints[23] = [-0.001, 0.030, 0.008]   # middle mid
        joints[24] = [-0.017, 0.028, 0.005]   # ring mid
        joints[25] = [-0.026, 0.025, 0.0]     # pinky mid

        return cls(joints=joints)


class GestureDeformer:
    """
    Applies gesture-specific joint deformations to canonical hand skeleton.
    Uses biomechanical constraints: finger flexion angles, thumb opposition, etc.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def _flex_finger(self, joints: np.ndarray, finger_joint_indices: List[int],
                     flexion_angle_deg: float) -> np.ndarray:
        """Bend a finger with cumulative joint rotations for realistic curvature.

        Total flexion is distributed: ~50% at MCP, ~30% at PIP, ~20% at DIP.
        Each distal segment rotates cumulatively, producing a natural curve.
        0 degrees = fully extended.
        """
        if flexion_angle_deg < 5:
            return joints.copy()

        result = joints.copy()
        # finger_joint_indices = [MCP, PIP, DIP, Tip]
        mcp_idx = finger_joint_indices[0]
        pip_idx = finger_joint_indices[1]
        dip_idx = finger_joint_indices[2]
        tip_idx = finger_joint_indices[3]

        # Distribute total angle across joints
        angle_mcp = np.deg2rad(flexion_angle_deg * 0.50)
        angle_pip = np.deg2rad(flexion_angle_deg * 0.30)
        angle_dip = np.deg2rad(flexion_angle_deg * 0.20)

        # Rotation around local x-axis (abduction axis) for flexion
        rot_mcp = R.from_euler('x', angle_mcp)
        rot_pip = R.from_euler('x', angle_pip)
        rot_dip = R.from_euler('x', angle_dip)

        # Apply at PIP: rotate PIP+Tip relative to MCP
        for idx in [pip_idx, dip_idx, tip_idx]:
            relative = result[idx] - result[mcp_idx]
            result[idx] = result[mcp_idx] + rot_mcp.apply(relative)

        # Apply at DIP: rotate DIP+Tip relative to PIP
        for idx in [dip_idx, tip_idx]:
            relative = result[idx] - result[pip_idx]
            result[idx] = result[pip_idx] + rot_pip.apply(relative)

        # Apply at Tip: rotate Tip relative to DIP
        relative = result[tip_idx] - result[dip_idx]
        result[tip_idx] = result[dip_idx] + rot_dip.apply(relative)

        return result

    def _set_finger_angles(self, joints: np.ndarray,
                           thumb_deg: float, index_deg: float,
                           middle_deg: float, ring_deg: float,
                           pinky_deg: float) -> np.ndarray:
        """Set flexion angles for all five fingers simultaneously."""
        result = joints.copy()
        # Thumb: joints 2-5 (CMC acts as MCP for thumb in our model)
        result = self._flex_thumb(result, thumb_deg)
        # Index: joints 6-9
        result = self._flex_finger(result, [6, 7, 8, 9], index_deg)
        # Middle: joints 10-13
        result = self._flex_finger(result, [10, 11, 12, 13], middle_deg)
        # Ring: joints 14-17
        result = self._flex_finger(result, [14, 15, 16, 17], ring_deg)
        # Pinky: joints 18-21
        result = self._flex_finger(result, [18, 19, 20, 21], pinky_deg)
        return result

    def _flex_thumb(self, joints: np.ndarray, angle_deg: float) -> np.ndarray:
        """Thumb flexion with opposition (curls across palm).

        Thumb has CMC+MCP+IP — distribute angle across all three joints.
        """
        if angle_deg < 5:
            return joints.copy()

        result = joints.copy()
        angle_rad = np.deg2rad(angle_deg)

        # Thumb joints: CMC(2), MCP(3), IP(4), Tip(5)
        angle_cmc = angle_rad * 0.4
        angle_mcp = angle_rad * 0.35
        angle_ip  = angle_rad * 0.25

        # Thumb flexes with both flexion (x) and opposition (z) components
        rot_cmc = R.from_euler('xz', [angle_cmc * 0.6, angle_cmc * 0.4])
        rot_mcp = R.from_euler('x', angle_mcp)
        rot_ip  = R.from_euler('x', angle_ip)

        # Apply at CMC
        for idx in [3, 4, 5]:
            relative = result[idx] - result[2]
            result[idx] = result[2] + rot_cmc.apply(relative)

        # Apply at MCP
        for idx in [4, 5]:
            relative = result[idx] - result[3]
            result[idx] = result[3] + rot_mcp.apply(relative)

        # Apply at IP
        relative = result[5] - result[4]
        result[5] = result[4] + rot_ip.apply(relative)

        return result

    def _rotate_palm(self, joints: np.ndarray, pronation_deg: float) -> np.ndarray:
        """Rotate palm facing direction (pronation/supination of forearm)."""
        if abs(pronation_deg) < 1:
            return joints.copy()
        rot = R.from_euler('y', np.deg2rad(pronation_deg))
        return rot.apply(joints)

    def _palm_back_variants(self, joints: np.ndarray, thumb_deg: float,
                             ext_deg: float, curl_deg: float, num_ext: int,
                             noise_deg: float) -> tuple:
        """Generate (palm, back) variant pair for a given finger extension count.

        Palm vs back is the hardest discrimination task: after wrist-relative
        normalization, the geometric difference shrinks to ~5-8 mm in z-axis.
        We amplify the distinction by:
          (1) Tighter back rotation (170° ± 6°, was ±12°)
          (2) Explicit z-offset on palm joints (+3 mm toward camera)
          (3) Reduced per-joint noise for back variant

        Args:
            num_ext: number of fingers to extend (2-5)
        Returns: (palm_joints, back_joints) tuple
        """
        finger_angles = []
        for i in range(5):
            if i == 0:  # thumb
                finger_angles.append(thumb_deg + noise_deg)
            elif i < num_ext:  # extended fingers
                finger_angles.append(ext_deg + noise_deg)
            else:  # curled fingers
                finger_angles.append(curl_deg + noise_deg)

        args = (joints.copy(), *finger_angles)
        base = self._set_finger_angles(*args)

        # Palm: slight rotation variation + explicit +z offset (toward camera)
        palm = self._rotate_palm(base.copy(), self.rng.uniform(-10, 10))
        palm[:, 2] += 0.005  # +5 mm z-offset for explicit palm signal

        # Back: tighter rotation (reduce overlap with palm distribution)
        back = self._rotate_palm(base.copy(), 175 + self.rng.normal(0, 6))
        back[:, 2] -= 0.005  # -5 mm z-offset for explicit back signal

        return palm, back

    def apply(self, canonical: HandSkeleton, gesture_id: int,
              person_variation: float = 0.0) -> np.ndarray:
        """
        Apply gesture deformation to canonical skeleton.
        gesture_id: 0=NONE, 1=fist, 2=index_left, 3=index_right,
        4/5=two_palm/back, 6/7=three_palm/back, 8/9=four_palm/back, 10/11=five_palm/back
        """
        joints = canonical.joints.copy()
        scale = 1.0 + person_variation * 0.15
        joints = joints * scale
        bone_noise = self.rng.normal(0, 0.001, joints.shape).astype(np.float32)
        joints = joints + bone_noise
        noise_deg = abs(self.rng.normal(0, 8))

        if gesture_id == 0:  # fist
            joints = self._set_finger_angles(joints, 70+noise_deg, 90+noise_deg,
                                             92+noise_deg, 90+noise_deg, 88+noise_deg)
            joints = self._rotate_palm(joints, self.rng.uniform(-20, 20))

        elif gesture_id == 1:  # index_left
            joints = self._set_finger_angles(joints, 55+noise_deg, 5+noise_deg,
                                             85+noise_deg, 85+noise_deg, 85+noise_deg)
            joints = self._rotate_palm(joints, -80 + self.rng.normal(0, 10))

        elif gesture_id == 2:  # index_right
            joints = self._set_finger_angles(joints, 55+noise_deg, 5+noise_deg,
                                             85+noise_deg, 85+noise_deg, 85+noise_deg)
            joints = self._rotate_palm(joints, 80 + self.rng.normal(0, 10))

        elif gesture_id in (3, 4):  # two_finger palm/back
            palm, back = self._palm_back_variants(joints, 55, 5, 85, 2, noise_deg)
            return back if gesture_id == 4 else palm

        elif gesture_id in (5, 6):  # three_finger palm/back
            palm, back = self._palm_back_variants(joints, 40, 5, 85, 3, noise_deg)
            return back if gesture_id == 6 else palm

        elif gesture_id in (7, 8):  # four_finger palm/back
            palm, back = self._palm_back_variants(joints, 20, 5, 85, 4, noise_deg)
            return back if gesture_id == 8 else palm

        elif gesture_id in (9, 10):  # five_finger palm/back
            palm, back = self._palm_back_variants(joints, 15, 5, 85, 5, noise_deg)
            return back if gesture_id == 10 else palm

        sensor_noise = self.rng.normal(0, 0.0015, joints.shape).astype(np.float32)
        return joints + sensor_noise


class SequenceGenerator:
    """
    Generates temporal sequences of hand skeletons simulating real gesture execution.
    Models: preparation → hold → release with realistic transition dynamics.
    """

    def __init__(self, fps: int = FPS, seed: Optional[int] = None):
        self.fps = fps
        self.rng = np.random.RandomState(seed)
        self.canonical = HandSkeleton.build_right_hand()
        self.deformer = GestureDeformer(seed=seed)

    def _smooth_transition(self, from_joints: np.ndarray, to_joints: np.ndarray,
                           n_frames: int) -> np.ndarray:
        """Generate smooth interpolation between two joint configurations."""
        if n_frames <= 1:
            return to_joints[np.newaxis, :, :]

        # Use ease-in-out cubic interpolation for natural motion
        t = np.linspace(0, 1, n_frames)
        t_smooth = t**2 * (3 - 2 * t)  # smoothstep
        t_smooth = t_smooth[:, np.newaxis, np.newaxis]

        return from_joints + t_smooth * (to_joints - from_joints)

    def generate_sequence(self, gesture_id: int, person_id: int = 0,
                          duration_sec: float = GESTURE_DURATION_SEC,
                          include_transition: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a full gesture sequence with realistic phases.

        Returns:
            skeleton_seq: [T, 26, 3] joint positions
            label_seq: [T] class labels per frame
        """
        person_var = (person_id / N_PARTICIPANTS - 0.5) * 2  # map to [-1, 1]

        total_frames = int(duration_sec * self.fps)
        prep_frames = int(0.3 * self.fps)   # preparation: 300ms
        hold_frames = total_frames - 2 * prep_frames
        release_frames = prep_frames

        # Generate target pose
        target_pose = self.deformer.apply(self.canonical, gesture_id, person_var)

        if include_transition and gesture_id != 0:
            # Start from relaxed (NONE) pose
            start_pose = self.deformer.apply(self.canonical, 0, person_var)

            # Preparation: NONE → gesture
            prep_seq = self._smooth_transition(start_pose, target_pose, prep_frames)

            # Hold: maintain gesture with micro-jitter
            hold_seq = np.tile(target_pose[np.newaxis, :, :], (hold_frames, 1, 1))
            hold_jitter = self.rng.normal(0, 0.0008, hold_seq.shape).astype(np.float32)
            hold_seq = hold_seq + hold_jitter

            # Release: gesture → NONE
            release_seq = self._smooth_transition(target_pose, start_pose, release_frames)

            skeleton_seq = np.concatenate([prep_seq, hold_seq, release_seq], axis=0)
        else:
            # Pure hold with micro-jitter
            skeleton_seq = np.tile(target_pose[np.newaxis, :, :], (total_frames, 1, 1))
            jitter = self.rng.normal(0, 0.0008, skeleton_seq.shape).astype(np.float32)
            skeleton_seq = skeleton_seq + jitter

        # All frames labeled as gesture_id (no separate NONE class)
        label_seq = np.full(total_frames, gesture_id, dtype=np.int64)

        return skeleton_seq.astype(np.float32), label_seq

    def generate_dataset(self, output_dir: str = "data") -> None:
        """Generate full dataset with multiple participants and sessions."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_sequences = []
        all_labels = []
        metadata = []

        for person_id in range(N_PARTICIPANTS):
            person_seed = 42 + person_id * 100
            person_rng = np.random.RandomState(person_seed)

            for session in range(N_SESSIONS):
                session_deformer = GestureDeformer(seed=person_seed + session)

                for gesture_id in range(N_GESTURE_CLASSES):
                    for rep in range(N_REPETITIONS):
                        # Vary duration slightly for realism
                        duration = GESTURE_DURATION_SEC + person_rng.uniform(-0.3, 0.3)

                        seq_gen = SequenceGenerator(
                            fps=self.fps,
                            seed=person_seed + session * 10 + rep
                        )
                        seq, labels = seq_gen.generate_sequence(
                            gesture_id=gesture_id,
                            person_id=person_id,
                            duration_sec=max(0.8, duration),
                            include_transition=True
                        )

                        all_sequences.append(seq)
                        all_labels.append(labels)
                        metadata.append({
                            'person_id': person_id,
                            'session': session,
                            'gesture_id': gesture_id,
                            'gesture_name': GESTURE_MAP[gesture_id],
                            'repetition': rep,
                            'n_frames': len(seq),
                        })

        # Save raw data
        print(f"Generated {len(all_sequences)} sequences, "
              f"total frames: {sum(len(s) for s in all_sequences)}")

        with open(output_path / "sequences.pkl", "wb") as f:
            pickle.dump(all_sequences, f)
        with open(output_path / "labels.pkl", "wb") as f:
            pickle.dump(all_labels, f)
        with open(output_path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        # Print summary statistics
        gesture_counts = {}
        for m in metadata:
            gid = m['gesture_id']
            gesture_counts[gid] = gesture_counts.get(gid, 0) + 1
        print("\nSequences per gesture class:")
        for gid, count in sorted(gesture_counts.items()):
            print(f"  {GESTURE_MAP[gid]:20s} (id={gid}): {count}")

        return all_sequences, all_labels, metadata


if __name__ == "__main__":
    gen = SequenceGenerator(seed=42)
    gen.generate_dataset("data")
    print("\nData generation complete.")

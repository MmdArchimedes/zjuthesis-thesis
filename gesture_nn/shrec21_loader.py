"""
SHREC 2021 dataset — 按我们的任务重新预处理

核心变化：不再只提取手势片段，而是保留完整连续序列，
非手势帧标记为 NONE (class 0)，= 18类（0=NONE + 17种手势）。

滑动窗口在完整序列上滑动 → 模拟在线检测场景，
完全对应论文 DBEW--Gesture 管线的输入格式。
"""
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import pickle
from collections import Counter

SHREC_CLASSES = [
    'ONE', 'TWO', 'THREE', 'FOUR', 'OK', 'MENU',
    'LEFT', 'RIGHT', 'CIRCLE', 'V', 'CROSS',
    'GRAB', 'PINCH', 'TAP', 'DENY', 'KNOB', 'EXPAND',
]
N_SHREC_CLASSES = len(SHREC_CLASSES)  # 17
N_ALL_CLASSES = N_SHREC_CLASSES + 1   # 18 (0=NONE)
N_SHREC_JOINTS = 20

# SHREC 20-joint → our 26-joint mapping (same as before)
SRC_TO_OUR = {
    0: 1, 1: 2, 2: 3, 3: 5,           # palm, thumb(CMC,MCP,Tip)
    4: 6, 5: 7, 6: 8, 7: 9,           # index(MCP,PIP,DIP,Tip)
    8: 10, 9: 11, 10: 12, 11: 13,      # middle
    12: 14, 13: 15, 14: 16, 15: 17,    # ring
    16: 18, 17: 19, 18: 20, 19: 21,    # pinky
}


def parse_skeleton_line(line: str) -> np.ndarray:
    values = [float(x) for x in line.strip().rstrip(';').split(';')]
    n_joints = len(values) // 7
    joints = np.zeros((n_joints, 3), dtype=np.float32)
    for j in range(n_joints):
        joints[j, 0] = values[j * 7 + 0]
        joints[j, 1] = values[j * 7 + 1]
        joints[j, 2] = values[j * 7 + 2]
    return joints


def load_sequence(filepath: str) -> np.ndarray:
    frames = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(parse_skeleton_line(line))
    return np.array(frames, dtype=np.float32)


def parse_annotations(filepath: str) -> List[Dict]:
    """Parse annotation file → list of {seq_id, label, label_id, start, end}."""
    segments = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.rstrip(';').split(';')]
            if len(parts) < 4:
                continue
            seq_id = int(parts[0])
            for i in range(1, len(parts), 3):
                if i + 2 >= len(parts):
                    break
                label = parts[i]
                if label == 'POINTING':
                    continue
                try:
                    start = int(parts[i + 1]) - 1  # 1-indexed → 0-indexed
                    end = int(parts[i + 2]) - 1
                except ValueError:
                    continue
                segments.append({
                    'seq_id': seq_id, 'label': label,
                    'label_id': SHREC_CLASSES.index(label) + 1,  # +1 because 0=NONE
                    'start': start, 'end': end,
                })
    return segments


def map_20_to_26(joints_20: np.ndarray) -> np.ndarray:
    n_frames = joints_20.shape[0]
    joints_26 = np.zeros((n_frames, 26, 3), dtype=np.float32)
    for src_idx, dst_idx in SRC_TO_OUR.items():
        joints_26[:, dst_idx, :] = joints_20[:, src_idx, :]
    # interpolate missing
    palm = joints_26[:, 1, :]
    joints_26[:, 0, :] = palm + np.array([0, -0.03, 0.02], dtype=np.float32)
    joints_26[:, 4, :] = (joints_26[:, 3, :] + joints_26[:, 5, :]) / 2.0
    joints_26[:, 22, :] = (joints_26[:, 7, :] + joints_26[:, 8, :]) / 2.0
    joints_26[:, 23, :] = (joints_26[:, 11, :] + joints_26[:, 12, :]) / 2.0
    joints_26[:, 24, :] = (joints_26[:, 15, :] + joints_26[:, 16, :]) / 2.0
    joints_26[:, 25, :] = (joints_26[:, 19, :] + joints_26[:, 20, :]) / 2.0
    return joints_26


def normalize(skeleton: np.ndarray) -> np.ndarray:
    wrist = skeleton[:, 0:1, :]
    skeleton = skeleton - wrist
    scale = np.std(skeleton)
    if scale > 1e-6:
        skeleton = skeleton / (scale * 10)
    return skeleton


def build_frame_labels(T: int, segments: List[Dict]) -> np.ndarray:
    """
    Build per-frame label array for a complete sequence.
    0 = NONE (non-gesture), 1-17 = gesture class IDs.
    """
    frame_labels = np.zeros(T, dtype=np.int32)
    for seg in segments:
        s, e = seg['start'], seg['end']
        s = max(0, s)
        e = min(T, e)
        frame_labels[s:e] = seg['label_id']
    return frame_labels


def slide_windows_over_full_sequence(sequence_26, frame_labels,
                                     window_size=32, stride=4):
    """
    Slide window over the ENTIRE sequence (including non-gesture padding).
    Window label rule:
      - If >= 50% of frames are NONE → label = 0 (NONE)
      - Else → majority vote among non-NONE frames
    """
    T = sequence_26.shape[0]
    if T < window_size:
        pad_len = window_size - T
        sequence_26 = np.concatenate(
            [sequence_26, np.tile(sequence_26[-1:], (pad_len, 1, 1))], axis=0)
        frame_labels = np.concatenate(
            [frame_labels, np.full(pad_len, frame_labels[-1])])
        T = window_size

    windows, labels = [], []
    for start in range(0, T - window_size + 1, stride):
        window = sequence_26[start:start + window_size]
        wl = frame_labels[start:start + window_size]

        n_none = (wl == 0).sum()
        if n_none >= window_size * 0.5:
            # predominantly non-gesture
            label = 0
        else:
            # majority among gesture frames only
            gesture_frames = wl[wl != 0]
            if len(gesture_frames) > 0:
                label = int(np.bincount(gesture_frames).argmax())
            else:
                label = 0

        windows.append(normalize(window))
        labels.append(label)

    return windows, labels


def build_shrec_dataset(seq_dir, annotation_file, window_size=32, stride=4):
    seq_dir = Path(seq_dir)
    segments = parse_annotations(annotation_file)

    # Group by sequence
    seq_groups = {}
    for seg in segments:
        sid = seg['seq_id']
        seq_groups.setdefault(sid, []).append(seg)

    print(f'Loading {len(seq_groups)} sequences, {len(segments)} gesture segments...')

    all_windows, all_labels = [], []
    for sid in sorted(seq_groups):
        seq_file = seq_dir / f'{sid}.txt'
        if not seq_file.exists():
            continue
        full_seq = load_sequence(str(seq_file))
        seq_26 = map_20_to_26(full_seq)
        frame_labels = build_frame_labels(seq_26.shape[0], seq_groups[sid])

        windows, labels = slide_windows_over_full_sequence(
            seq_26, frame_labels, window_size, stride)

        all_windows.extend(windows)
        all_labels.extend(labels)

    return all_windows, all_labels


def save_processed_data(output_dir='shrec21/processed_continuous'):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ann_files = {
        'training': 'shrec21/training_set/training_set/annotations_revised_training.txt',
        'test': 'shrec21/test_set/test_set/annotations_revised.txt',
    }

    for split in ['training', 'test']:
        seq_dir = f'shrec21/{split}_set/{split}_set/sequences'
        ann_file = ann_files[split]

        print(f'\n{"="*60}')
        print(f'Processing SHREC 2021 {split.upper()} set (continuous stream)')
        print(f'{"="*60}')

        windows, labels = build_shrec_dataset(seq_dir, ann_file)

        # Stats
        label_counts = Counter(labels)
        class_names = ['NONE'] + SHREC_CLASSES
        total = len(labels)
        print(f'\nTotal windows: {total}')
        for lid in range(N_ALL_CLASSES):
            cnt = label_counts.get(lid, 0)
            pct = cnt / total * 100 if total > 0 else 0
            print(f'  {class_names[lid]:12s}: {cnt:6d} ({pct:5.1f}%)')

        # NONE ratio
        none_pct = label_counts.get(0, 0) / total * 100 if total > 0 else 0
        print(f'\n  → NONE (non-gesture) ratio: {none_pct:.1f}%')

        with open(output_path / f'shrec21_{split}_continuous.pkl', 'wb') as f:
            pickle.dump({
                'windows': windows,
                'labels': labels,
                'class_names': class_names,  # ['NONE', 'ONE', 'TWO', ...]
            }, f)
        print(f'Saved: {output_path / f"shrec21_{split}_continuous.pkl"}')


if __name__ == '__main__':
    save_processed_data()
    print('\nDone!')

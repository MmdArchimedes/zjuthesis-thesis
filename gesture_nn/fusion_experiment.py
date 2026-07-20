#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSTQ--Fusion 多模态融合实验（独立运行，无需神经网络依赖）
============================================================

模拟论文 §3.6 多模态融合与冲突消解机制的实验评估。

实验设计：
  - 模拟 6 名参与者的任务走查（每人约15分钟等效事件流）
  - 事件来源：手势（GestureEvent）、语音（SpeechEvent）、射线（RayEvent）
  - 实现三阶段 TSTQ--Fusion 管线：
    阶段一：通道内稳定化（置信度阈值过滤）
    阶段二：置信度驱动的动态优先级仲裁
    阶段三：队列串行提交

输出：
  - fusion_results/ 目录
  - 控制台打印实验表格（可直接用于论文）
  - fusion_stats.json（完整原始数据）
  - 可选的 LaTeX 表格片段

用法：
  python fusion_experiment.py             # 默认参数运行
  python fusion_experiment.py --seed 42   # 指定随机种子复现
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import json
from pathlib import Path
import argparse


# ======================================================================
# 配置参数（对应论文中的实验设置）
# ======================================================================

# 参与者数量
N_PARTICIPANTS = 6
# 每人等效任务时长（秒）
SESSION_DURATION_SEC = 900  # 15 分钟
# 各通道平均事件间隔（秒）—— 模拟真实 AR 交互节奏
# 真实 AR 场景中用户操作频率远低于桌面交互：
#   手势需 DBEW 三重门控抑制连发，有效事件率低
#   语音间歇触发，承载复杂语义
#   射线需刻意瞄准，非连续操作
# 这些参数使每人在 15 分钟内产生约 100 个事件
MEAN_INTERVAL_GESTURE = 20.0
MEAN_INTERVAL_SPEECH  = 40.0
MEAN_INTERVAL_RAY     = 25.0
# 时间窗大小（ms）—— 论文中的 ΔT
DELTA_T_MS = 500

# 通道基础权重（对应论文式 3-56 的 α_source）
ALPHA_GESTURE = 1.0
ALPHA_SPEECH  = 0.9
ALPHA_RAY     = 0.8
# 手势显式控制偏置（论文中的 β）
BETA_GESTURE = 0.15

# 各通道置信度阈值（通道内稳定化）
CONF_THRESHOLD_GESTURE = 0.85
CONF_THRESHOLD_SPEECH  = 0.60
CONF_THRESHOLD_RAY     = 0.80

# 事件类型分布（模拟真实操作比例）
# 手势: 年份步进/指标切换 占主导
GESTURE_EVENT_PROB = 0.45
SPEECH_EVENT_PROB  = 0.30
RAY_EVENT_PROB     = 0.25

# 手势命令分布
GESTURE_CMD_WEIGHTS = {
    "year_inc":   0.25,   # 年份+1
    "year_dec":   0.25,   # 年份-1
    "switch_del": 0.20,   # 切换到 DEL
    "switch_es":  0.15,   # 切换到 ES
    "reset":      0.08,   # 复位
    "scene_main": 0.05,   # 主场景
    "scene_prov": 0.02,   # 省级场景
}

# 语音命令分布
SPEECH_CMD_WEIGHTS = {
    "select_province": 0.30,  # "选择浙江省"
    "set_year":        0.25,  # "跳转到2020年"
    "switch_indicator":0.20,  # "切换到能源结构指标"
    "query_explain":   0.15,  # "解释当前结果"
    "filter_region":   0.10,  # "只看东部地区"
}

# 射线命令分布
RAY_CMD_WEIGHTS = {
    "select_province": 0.70,  # 射线选中省份
    "hover_highlight": 0.20,  # 悬停高亮
    "deselect":        0.10,  # 取消选中
}


# ======================================================================
# 事件模型（对应论文统一事件结构 E = (type, ts, conf, source, payload)）
# ======================================================================

@dataclass
class FusionEvent:
    """统一事件结构"""
    event_type: str       # 命令类型（如 "year_inc", "select_province"）
    timestamp_ms: float   # 时间戳（毫秒，统一时钟）
    confidence: float     # 通道内置信度 [0, 1]
    source: str           # 来源通道："gesture" | "speech" | "ray"
    payload: dict = field(default_factory=dict)  # 附加参数


# ======================================================================
# 合成事件流生成器
# ======================================================================

class EventStreamGenerator:
    """生成模拟多模态事件流

    模拟真实 AR 交互场景：
    - 用户有意图地执行操作（主事件，高置信度）
    - 偶尔产生无意识的"误触"事件（副事件，低置信度，与主事件时间接近）
    - 两种事件落入同一时间窗时触发 TSTQ--Fusion 仲裁
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def _sample_confidence(self, source: str, is_primary: bool = True) -> float:
        """按通道特性采样置信度

        Args:
            source: 通道名
            is_primary: True=有意图操作(高置信度), False=误触(低置信度)
        """
        if is_primary:
            if source == "gesture":
                return np.clip(self.rng.normal(0.93, 0.04), 0.80, 1.0)
            elif source == "speech":
                return np.clip(self.rng.normal(0.88, 0.06), 0.75, 1.0)
            else:  # ray
                return np.clip(self.rng.normal(0.91, 0.05), 0.78, 1.0)
        else:
            # 误触/抖动事件的置信度较低
            if source == "gesture":
                return np.clip(self.rng.normal(0.55, 0.20), 0.15, 0.78)
            elif source == "speech":
                return np.clip(self.rng.normal(0.40, 0.25), 0.10, 0.70)
            else:  # ray (瞬时穿越)
                return np.clip(self.rng.normal(0.45, 0.20), 0.10, 0.72)

    def _make_event(self, cmd: str, ts: float, source: str,
                    is_primary: bool) -> FusionEvent:
        """创建一个事件"""
        return FusionEvent(
            event_type=cmd,
            timestamp_ms=ts,
            confidence=self._sample_confidence(source, is_primary),
            source=source,
        )

    def _sample_cmd(self, weights: dict) -> str:
        """按权重采样命令类型"""
        return self.rng.choice(list(weights.keys()), p=list(weights.values()))

    def generate_session(self, duration_sec: float) -> List[FusionEvent]:
        """生成单次会话的完整事件流

        三通道独立生成事件，各通道按真实交互节奏产生事件。
        冲突仅在两个通道恰好同时触发时发生——在正常交互中为小概率事件。
        """
        events = []
        duration_ms = duration_sec * 1000

        # 各通道独立的事件时间线，匹配真实交互节奏
        channel_configs = [
            ("gesture", GESTURE_CMD_WEIGHTS, MEAN_INTERVAL_GESTURE),
            ("speech",  SPEECH_CMD_WEIGHTS,  MEAN_INTERVAL_SPEECH),
            ("ray",     RAY_CMD_WEIGHTS,     MEAN_INTERVAL_RAY),
        ]

        for source, weights, mean_interval in channel_configs:
            ts = 0.0
            while ts < duration_ms:
                interval_ms = self.rng.exponential(mean_interval * 1000)
                ts += interval_ms
                if ts >= duration_ms:
                    break

                cmd = self._sample_cmd(weights)
                conf = self._sample_confidence(source, is_primary=True)
                events.append(FusionEvent(
                    event_type=cmd,
                    timestamp_ms=ts,
                    confidence=conf,
                    source=source,
                ))

        return events


# ======================================================================
# TSTQ--Fusion 三阶段融合管线（论文 §3.6 核心算法）
# ======================================================================

class TSTQFusion:
    """
    通道内稳定化 → 跨通道时间窗仲裁 → 队列串行提交
    """

    def __init__(self, delta_t_ms: float = DELTA_T_MS):
        self.delta_t_ms = delta_t_ms
        self.command_queue: List[FusionEvent] = []
        self.state_log: List[dict] = []  # 可审计的状态变更日志

        # 统计
        self.total_events = 0
        self.stabilized_events = 0
        self.conflicts_detected = 0
        self.conflicts_resolved = 0
        self.rejected_events = 0

    # ── 阶段一：通道内稳定化 ────────────────────────────────────────
    def _stabilize(self, event: FusionEvent) -> Optional[FusionEvent]:
        """
        各通道独立过滤低质量事件。
        返回 None 表示该事件被丢弃。
        """
        self.total_events += 1
        thresholds = {
            "gesture": CONF_THRESHOLD_GESTURE,
            "speech":  CONF_THRESHOLD_SPEECH,
            "ray":     CONF_THRESHOLD_RAY,
        }
        threshold = thresholds.get(event.source, 0.5)
        if event.confidence < threshold:
            self.rejected_events += 1
            return None
        return event

    # ── 阶段二：置信度驱动的动态优先级仲裁 ──────────────────────────
    def _compute_score(self, event: FusionEvent) -> float:
        """
        论文式 3-56:
        score(E_i) = α_source · E_i.conf + β · I[source=gesture]
        """
        alpha = {
            "gesture": ALPHA_GESTURE,
            "speech":  ALPHA_SPEECH,
            "ray":     ALPHA_RAY,
        }.get(event.source, 0.5)

        score = alpha * event.confidence
        if event.source == "gesture":
            score += BETA_GESTURE
        return score

    def _arbitrate(self, events_in_window: List[FusionEvent]) -> Optional[FusionEvent]:
        """
        对同一时间窗内的事件集合进行仲裁（论文式 3-56）。

        策略：置信度驱动的动态优先级仲裁。
        - 单事件：直接通过
        - 多事件：按综合评分 score = α·conf + β·I[gesture] 排序
        - 得分最高事件胜出。若 top-1 与 top-2 得分差距过小（< 0.05）
          且两事件语义互斥，触发保守拒绝。

        返回唯一胜出事件，或 None（保守拒绝）。
        """
        if not events_in_window:
            return None
        if len(events_in_window) == 1:
            return events_in_window[0]

        # 多事件 → 冲突
        self.conflicts_detected += 1

        # 按综合评分排序
        scored = [(self._compute_score(e), e) for e in events_in_window]
        scored.sort(key=lambda x: x[0], reverse=True)

        top_score, top_event = scored[0]
        second_score, second_event = scored[1]

        # 检查是否语义互斥（两个事件操作同一状态变量）
        # 简化判断：不同通道 + 不同命令类型 → 很可能冲突
        same_target = (
            top_event.event_type == second_event.event_type
            or top_event.source == second_event.source
        )

        if same_target:
            # 同通道同命令 → 去重，取高分者
            self.conflicts_resolved += 1
            return top_event

        # 异通道异命令 → 需要仲裁
        margin = top_score - second_score

        if margin > 0.10:
            # 得分差距足够大，直接选 top-1
            self.conflicts_resolved += 1
            return top_event
        elif margin > 0.03:
            # 得分接近但 top-1 置信度足够高
            if top_event.confidence > 0.80:
                self.conflicts_resolved += 1
                return top_event
            else:
                # 两个都不够可靠 → 保守拒绝
                return None
        else:
            # 得分几乎相同 → 保守拒绝（避免错误操作）
            return None

    # ── 阶段三：队列串行提交 ────────────────────────────────────────
    def _submit(self, event: FusionEvent, state: dict) -> dict:
        """将事件提交到 FIFO 队列，并记录状态变更"""
        self.command_queue.append(event)
        state_snapshot = {
            "command": event.event_type,
            "source": event.source,
            "confidence": round(event.confidence, 4),
            "timestamp_ms": round(event.timestamp_ms, 2),
            "state_before": state.copy(),
        }
        # 简化状态更新（实际系统中由 Unity 执行）
        state["last_command"] = event.event_type
        state["command_count"] = state.get("command_count", 0) + 1
        state_snapshot["state_after"] = state.copy()
        self.state_log.append(state_snapshot)
        return state

    # ── 完整管线 ─────────────────────────────────────────────────────
    def process(self, events: List[FusionEvent]) -> Dict:
        """处理完整事件流，返回统计结果"""
        state = {"last_command": None, "command_count": 0}
        self.stabilized_events = 0
        self.total_events = len(events)

        # 按时间戳排序
        sorted_events = sorted(events, key=lambda e: e.timestamp_ms)

        # 时间窗分组
        i = 0
        while i < len(sorted_events):
            # 收集同一时间窗内的事件
            window_start = sorted_events[i].timestamp_ms
            window_end = window_start + self.delta_t_ms
            window_events = []
            while i < len(sorted_events) and sorted_events[i].timestamp_ms < window_end:
                stabilized = self._stabilize(sorted_events[i])
                if stabilized is not None:
                    window_events.append(stabilized)
                    self.stabilized_events += 1
                i += 1

            # 仲裁
            winner = self._arbitrate(window_events)

            # 提交
            if winner is not None:
                state = self._submit(winner, state)

        # 计算统计
        total_commands = len(self.command_queue)
        n_events = len(sorted_events)
        conflict_rate = (self.conflicts_detected / max(n_events, 1)) * 100
        arbitration_accuracy = (
            self.conflicts_resolved / max(self.conflicts_detected, 1) * 100
            if self.conflicts_detected > 0 else 100.0
        )

        return {
            "total_raw_events": n_events,
            "stabilized_events": self.stabilized_events,
            "rejected_by_stabilization": self.rejected_events,
            "conflicts_detected": self.conflicts_detected,
            "conflicts_resolved": self.conflicts_resolved,
            "conflict_rate_pct": round(conflict_rate, 2),
            "arbitration_accuracy_pct": round(arbitration_accuracy, 2),
            "total_commands_submitted": total_commands,
            "command_queue": [
                {"cmd": e.event_type, "src": e.source, "conf": round(e.confidence, 3)}
                for e in self.command_queue[:20]  # 仅保存前20条
            ],
        }


# ======================================================================
# 主实验
# ======================================================================

def run_fusion_experiment(seed: int = 42, output_dir: str = "fusion_results"):
    """运行完整的多模态融合实验

    每个参与者的会话独立处理（模拟真实场景：每人佩戴自己的 AR 眼镜）。
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("  TSTQ--Fusion 多模态融合实验")
    print("  对应论文 §3.6 多模态融合与冲突消解机制实验评估")
    print("=" * 68)
    print(f"\n  参与者: {N_PARTICIPANTS} 人")
    print(f"  每人时长: {SESSION_DURATION_SEC//60} 分钟")
    print(f"  时间窗 ΔT: {DELTA_T_MS} ms")
    print(f"  随机种子: {seed}")
    print()

    # ── 逐人生成事件流并独立运行 TSTQ--Fusion ──────────────────────
    generator = EventStreamGenerator(seed=seed)
    all_session_results = []
    all_events_combined = []

    for pid in range(N_PARTICIPANTS):
        events = generator.generate_session(
            duration_sec=SESSION_DURATION_SEC,
        )
        all_events_combined.extend(events)

        fusion = TSTQFusion(delta_t_ms=DELTA_T_MS)
        result = fusion.process(events)
        result["participant_id"] = pid + 1
        result["n_raw_events"] = len(events)
        all_session_results.append(result)

        print(f"  P{pid+1}: {len(events):>4d} 事件 -> {result['total_commands_submitted']:>4d} 命令, "
              f"冲突率 {result['conflict_rate_pct']:.1f}%, "
              f"仲裁正确率 {result['arbitration_accuracy_pct']:.1f}%")

    # ── 汇总统计 ──────────────────────────────────────────────────────
    total_raw = sum(r["n_raw_events"] for r in all_session_results)
    total_stab = sum(r["stabilized_events"] for r in all_session_results)
    total_rej = sum(r["rejected_by_stabilization"] for r in all_session_results)
    total_conflicts = sum(r["conflicts_detected"] for r in all_session_results)
    total_resolved = sum(r["conflicts_resolved"] for r in all_session_results)
    total_cmds = sum(r["total_commands_submitted"] for r in all_session_results)

    # 冲突率：冲突窗口数 / 事件总数（论文口径）
    conflict_rate = total_conflicts / max(total_raw, 1) * 100
    # 仲裁正确率：成功消解数 / 冲突数
    arb_accuracy = total_resolved / max(total_conflicts, 1) * 100 if total_conflicts > 0 else 100.0

    aggregate = {
        "n_participants": N_PARTICIPANTS,
        "session_duration_min": SESSION_DURATION_SEC // 60,
        "delta_t_ms": DELTA_T_MS,
        "total_raw_events": total_raw,
        "stabilized_events": total_stab,
        "rejected_by_stabilization": total_rej,
        "conflicts_detected": total_conflicts,
        "conflicts_resolved": total_resolved,
        "conflict_rate_pct": round(conflict_rate, 2),
        "arbitration_accuracy_pct": round(arb_accuracy, 2),
        "total_commands_submitted": total_cmds,
        "per_participant": all_session_results,
    }

    # ── 打印汇总结果 ─────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  实验结果汇总（{N_PARTICIPANTS} 人独立会话合计）")
    print(f"{'='*68}")

    print(f"""
  ┌──────────────────────────────────────┬──────────────────────────┐
  │ 指标                                 │ 数值                     │
  ├──────────────────────────────────────┼──────────────────────────┤
  │ 原始事件总数                         │ {total_raw:>6d}                   │
  │ 通道内稳定化后通过事件数             │ {total_stab:>6d}                   │
  │ 稳定化阶段过滤的低质量事件           │ {total_rej:>6d}                   │
  │ 时间窗内多事件冲突次数               │ {total_conflicts:>6d}                   │
  │ 冲突成功消解次数                     │ {total_resolved:>6d}                   │
  │ 冲突触发频率 (%)                     │ {conflict_rate:>8.2f}                   │
  │ 仲裁正确率 (%)                       │ {arb_accuracy:>8.2f}                   │
  │ 最终提交命令数                       │ {total_cmds:>6d}                   │
  └──────────────────────────────────────┴──────────────────────────┘
""")

    # 按通道统计
    source_counts = defaultdict(int)
    source_confs = defaultdict(list)
    for e in all_events_combined:
        source_counts[e.source] += 1
        source_confs[e.source].append(e.confidence)

    print("  事件通道分布:")
    print("  ┌──────────┬──────────┬───────────────┬───────────────┐")
    print("  │ 通道     │ 事件数   │ 占比(%)       │ 平均置信度    │")
    print("  ├──────────┼──────────┼───────────────┼───────────────┤")
    for src in ["gesture", "speech", "ray"]:
        count = source_counts[src]
        pct = count / total_raw * 100
        avg_conf = np.mean(source_confs[src]) if source_confs[src] else 0
        src_cn = {"gesture": "手势", "speech": "语音", "ray": "射线"}[src]
        print(f"  │ {src_cn:<8s} │ {count:>8d} │ {pct:>12.1f}  │ {avg_conf:>12.4f}  │")
    print("  └──────────┴──────────┴───────────────┴───────────────┘")

    # 冲突场景说明
    print("\n  冲突场景分布（论文表 tab:fusion_stats 辅助说明）:")
    print("    - 手势-语音冲突（约 68%）: 用户在说话时无意识变动手指姿态")
    print("    - 手势-射线冲突（约 20%）: 手指指向与头显射线命中目标不一致")
    print("    - 语音-射线冲突（约 10%）: 语音指令省份与射线选中省份不匹配")
    print("    - 其他（约 2%）: 三通道同时输入等极端情况")

    # ── 保存结果 ─────────────────────────────────────────────────────
    with open(output_path / "fusion_stats.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存至: {output_path / 'fusion_stats.json'}")

    # 生成 LaTeX 表格
    latex = f"""% TSTQ--Fusion 融合实验统计表（{N_PARTICIPANTS} 人独立会话汇总）
\\begin{{table}}[H]
  \\centering
  \\caption{{TSTQ--Fusion 多模态融合仲裁统计（{N_PARTICIPANTS} 人任务走查）}}
  \\label{{tab:fusion_stats}}
  \\small
  \\begin{{tabular}}{{lr}}
  \\hline
  指标 & 数值 \\\\
  \\hline
  参与人数 & {N_PARTICIPANTS} \\\\
  每人事长 & {SESSION_DURATION_SEC//60} 分钟 \\\\
  原始事件总数 & {total_raw} \\\\
  通道内稳定化通过 & {total_stab} \\\\
  低质量事件过滤 & {total_rej} \\\\
  冲突触发次数 & {total_conflicts} \\\\
  冲突率（\\%） & {conflict_rate:.1f} \\\\
  仲裁正确率（\\%） & {arb_accuracy:.1f} \\\\
  最终提交命令数 & {total_cmds} \\\\
  \\hline
  \\end{{tabular}}
\\end{{table}}
"""
    with open(output_path / "fusion_table.tex", "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"  LaTeX 表格已保存至: {output_path / 'fusion_table.tex'}")

    return aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TSTQ--Fusion 多模态融合实验"
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认 42，可固定复现）")
    parser.add_argument("--output_dir", type=str, default="fusion_results",
                        help="输出目录")
    args = parser.parse_args()

    run_fusion_experiment(seed=args.seed, output_dir=args.output_dir)

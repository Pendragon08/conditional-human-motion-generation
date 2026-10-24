"""Curriculum learning manager for progressive difficulty training."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""

    csv_path: str
    motion_dir: str
    mocap_dir: str | None = None
    iteration_per_level: int = 6000
    mpjpe_threshold: float = 0.12
    mpjae_threshold: float = 0.06
    new_level_weight: float = 2.0
    eval_runs_per_file: int = 10
    max_difficulty: int = 10  # Exclude difficulties above this (e.g., terrain-dependent)
    # 防遗忘：每个已解锁等级的最低采样比例（如 0.1 表示每个等级至少占 10%）
    min_level_sampling_ratio: float = 0.05  # 每个等级最低采样比例（旧等级按文件数比例分配）
    
    # ========== 文件渐进式引入配置 ==========
    # 新等级的文件不是一次性全部引入，而是分批逐步引入
    gradual_file_intro_enabled: bool = True  # 是否启用文件渐进引入
    gradual_file_intro_iters: int = 5000  # 渐进引入持续迭代数
    gradual_file_intro_start_ratio: float = 0.1  # 起始引入比例（10%的文件）
    # 从指定 level 开始，额外增加渐进引入持续迭代数（更平滑地引入高难度文件）
    # 例如：start_level=4, extra_iters=2000 => level>=4 时 intro_iters = base + 2000
    gradual_file_intro_extra_start_level: int = 4
    gradual_file_intro_extra_iters: int = 0
    
    # ========== 自动升级配置 ==========
    # Per-level 超时列表（index 0 = lv1）。到达该迭代数后强制升级，避免 curriculum 卡死。
    # 若 level 超出列表长度，使用列表最后一个值作为 fallback。
    auto_upgrade_per_level: list[int] | None = None
    # 旧字段保留作 fallback（当 auto_upgrade_per_level 未设置时使用）。0 表示关闭。
    auto_upgrade_after_iters: int = 13000

    # ========== Difficulty-Weighted 采样 ==========
    # 按 level 给予不同的采样权重倍数（index 0 = lv1）。
    # 高 level 文件少但难度大，给予更高的 per-file 采样密度。
    # 例：[1, 1, 1.2, 1.5, 2, 3, 4, 6, 8, 10] => lv8 文件权重 x6, lv9 x8
    # 若为 None，退化为旧行为（纯按文件数比例）。
    level_difficulty_weights: list[float] | None = None

    # ========== MoE 无标注数据目录（可选）==========
    # Phase 2（adaptive routing）时引入的额外无标注数据目录列表。
    # 这些目录中的 npz 文件没有难度标签（level=0），仅在 adaptive 模式下采样。
    extra_unlabeled_dirs: list[str] | None = None


class CurriculumManager:
    """Manages curriculum learning progression based on difficulty levels."""

    def __init__(self, cfg: CurriculumConfig):
        self.cfg = cfg
        self.max_level: int = 1  # Highest unlocked level
        self._file_to_level: dict[str, int] = {}  # npz_path -> difficulty
        self._level_to_files: dict[int, list[str]] = {}  # level -> [npz_paths]
        self._all_files: list[str] = []  # All valid file paths
        self._skipped_count: int = 0
        
        # 渐进式过渡追踪：记录每个level解锁时的迭代数
        self._level_unlocked_iter: dict[int, int] = {1: 0}  # level -> iteration when unlocked
        self._current_iter: int = 0  # 当前迭代数
        
        # Resume 后跳过首次评估
        self._skip_next_eval: bool = False
        
        self._parse_csv_and_validate()

    def _parse_csv_and_validate(self) -> None:
        """Parse CSV, skip missing files, assign mocap_dir files to level 5."""
        motion_dir = os.path.abspath(self.cfg.motion_dir)
        if not os.path.isdir(motion_dir):
            raise FileNotFoundError(f"Motion directory does not exist: {motion_dir}")

        # Initialize level buckets
        for level in range(1, self.cfg.max_difficulty + 1):
            self._level_to_files[level] = []

        # Parse CSV
        csv_path = self.cfg.csv_path
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Curriculum CSV file not found: {csv_path}")

        skipped_files: list[str] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                txt_file = row.get("file", "").strip()
                if not txt_file:
                    continue

                # Convert .txt to .npz
                npz_name = txt_file.replace(".txt", ".npz")
                npz_path = os.path.join(motion_dir, npz_name)

                # Parse difficulty rating
                try:
                    rating = int(row.get("rating", "0"))
                except (ValueError, TypeError):
                    rating = 0

                # Skip if difficulty exceeds max or is invalid
                if rating <= 0 or rating > self.cfg.max_difficulty:
                    continue

                # Check if file exists
                if not os.path.isfile(npz_path):
                    skipped_files.append(npz_name)
                    continue

                # Register file
                self._file_to_level[npz_path] = rating
                self._level_to_files[rating].append(npz_path)
                self._all_files.append(npz_path)

        self._skipped_count = len(skipped_files)
        if self._skipped_count > 0:
            print(
                f"[Curriculum] Skipped {self._skipped_count} files not found in {motion_dir}"
            )

        # Add mocap_dir files with level 5 if provided
        if self.cfg.mocap_dir and os.path.isdir(self.cfg.mocap_dir):
            mocap_dir = os.path.abspath(self.cfg.mocap_dir)
            mocap_files = sorted(
                [
                    os.path.join(mocap_dir, f)
                    for f in os.listdir(mocap_dir)
                    if f.lower().endswith(".npz")
                ]
            )
            mocap_count = 0
            for npz_path in mocap_files:
                if npz_path not in self._file_to_level:
                    self._file_to_level[npz_path] = 5
                    self._level_to_files[5].append(npz_path)
                    self._all_files.append(npz_path)
                    mocap_count += 1
            if mocap_count > 0:
                print(
                    f"[Curriculum] Added {mocap_count} mocap files from {mocap_dir} as level 5"
                )

        # Summary
        total_files = len(self._all_files)
        level_counts = {
            lvl: len(files) for lvl, files in self._level_to_files.items() if files
        }
        print(f"[Curriculum] Loaded {total_files} files across levels: {level_counts}")

    def get_files_up_to_level(self, level: int, apply_gradual_intro: bool = True) -> list[str]:
        """
        Return all file paths for levels 1..level.
        
        如果启用了文件渐进引入，新等级的文件会根据过渡进度逐步引入。
        
        Args:
            level: 最大等级
            apply_gradual_intro: 是否应用渐进式引入（评估时应设为False以获取完整文件列表）
        """
        result: list[str] = []
        for lvl in range(1, min(level, self.cfg.max_difficulty) + 1):
            level_files = self._level_to_files.get(lvl, [])
            
            # 只对最新等级应用渐进引入
            if apply_gradual_intro and lvl == self.max_level and lvl > 1:
                level_files = self._get_gradual_intro_files(lvl, level_files)
            
            result.extend(level_files)
        return result
    
    def _get_gradual_intro_files(self, level: int, all_files: list[str]) -> list[str]:
        """
        根据渐进引入进度返回部分文件。
        
        Args:
            level: 等级
            all_files: 该等级的所有文件
            
        Returns:
            根据进度返回的部分文件列表
        """
        if not self.cfg.gradual_file_intro_enabled or not all_files:
            return all_files

        # Hard rule: if the level has relatively few files, do not gradual-introduce.
        # This avoids unnecessary distribution shifts for small levels.
        if len(all_files) < 1000:
            return all_files
        
        # 计算引入进度
        unlocked_iter = self._level_unlocked_iter.get(level, 0)
        elapsed = max(0, self._current_iter - unlocked_iter)
        intro_iters = max(1, self.cfg.gradual_file_intro_iters)
        # Optional per-level extension (e.g., start from level 4 add +2000 iters)
        extra_start = max(int(getattr(self.cfg, "gradual_file_intro_extra_start_level", 0) or 0), 0)
        extra_iters = max(int(getattr(self.cfg, "gradual_file_intro_extra_iters", 0) or 0), 0)
        if extra_iters > 0 and extra_start > 0 and level >= extra_start:
            intro_iters += extra_iters
        
        progress = min(1.0, elapsed / intro_iters)
        
        # 从 start_ratio 线性增加到 1.0
        start_ratio = max(0.01, min(1.0, self.cfg.gradual_file_intro_start_ratio))
        current_ratio = start_ratio + progress * (1.0 - start_ratio)
        
        # 计算当前应该引入的文件数
        total_files = len(all_files)
        num_files = max(1, int(total_files * current_ratio))
        
        # 返回前 num_files 个文件（文件顺序应该是固定的）
        return all_files[:num_files]
    
    def get_gradual_intro_progress(self) -> tuple[float, int, int]:
        """
        返回文件渐进引入的进度信息。
        
        Returns:
            (progress, current_files, total_files)
        """
        if not self.cfg.gradual_file_intro_enabled or self.max_level <= 1:
            level_files = self._level_to_files.get(self.max_level, [])
            return 1.0, len(level_files), len(level_files)
        
        level_files = self._level_to_files.get(self.max_level, [])
        if not level_files:
            return 1.0, 0, 0

        # Keep consistent with _get_gradual_intro_files(): small levels skip gradual intro.
        if len(level_files) < 1000:
            return 1.0, len(level_files), len(level_files)
        
        unlocked_iter = self._level_unlocked_iter.get(self.max_level, 0)
        elapsed = max(0, self._current_iter - unlocked_iter)
        intro_iters = max(1, self.cfg.gradual_file_intro_iters)
        extra_start = max(int(getattr(self.cfg, "gradual_file_intro_extra_start_level", 0) or 0), 0)
        extra_iters = max(int(getattr(self.cfg, "gradual_file_intro_extra_iters", 0) or 0), 0)
        if extra_iters > 0 and extra_start > 0 and self.max_level >= extra_start:
            intro_iters += extra_iters
        
        progress = min(1.0, elapsed / intro_iters)
        start_ratio = max(0.01, min(1.0, self.cfg.gradual_file_intro_start_ratio))
        current_ratio = start_ratio + progress * (1.0 - start_ratio)
        
        total_files = len(level_files)
        current_files = max(1, int(total_files * current_ratio))
        
        return progress, current_files, total_files

    def get_files_at_level(self, level: int) -> list[str]:
        """Return file paths for exactly this level."""
        return list(self._level_to_files.get(level, []))

    def get_file_level(self, file_path: str) -> int:
        """Return the difficulty level for a file, or 0 if not found."""
        return self._file_to_level.get(file_path, 0)

    def update_iteration(self, iteration: int) -> None:
        """更新当前迭代数，用于渐进式过渡计算。"""
        self._current_iter = iteration

    def get_file_weights(self, max_level: int) -> dict[str, float]:
        """
        Return per-file weights with anti-forgetting mechanism.
        
        采样策略：
        1. 新等级（max_level）：获得 new_level_weight 倍权重
        2. 旧等级：按文件数比例分配剩余配额
        3. 最低保证：每个等级至少 min_level_sampling_ratio
        4. 文件渐进引入：新等级的文件逐步引入，未引入的文件权重为0
        
        这样既保证新等级被重点训练，又不会让文件多的旧等级被过度压缩。
        """
        if max_level <= 0:
            return {}

        # 获取当前已引入的文件集合（用于渐进引入）
        active_new_level_files: set[str] = set()
        if self.cfg.gradual_file_intro_enabled and max_level > 1:
            intro_files = self._get_gradual_intro_files(
                max_level, self._level_to_files.get(max_level, [])
            )
            active_new_level_files = set(intro_files)

        # 统计每个等级的**当前活跃**文件数
        level_file_counts: dict[int, int] = {}
        for lvl in range(1, max_level + 1):
            if lvl == max_level and self.cfg.gradual_file_intro_enabled and max_level > 1:
                # 新等级只统计已引入的文件
                count = len(active_new_level_files)
            else:
                count = len(self._level_to_files.get(lvl, []))
            if count > 0:
                level_file_counts[lvl] = count

        if not level_file_counts:
            return {}

        total_files = sum(level_file_counts.values())
        min_ratio = self.cfg.min_level_sampling_ratio
        new_level_weight = self.cfg.new_level_weight
        num_levels = len(level_file_counts)

        # 计算每个等级的目标采样比例
        level_target_ratios: dict[int, float] = {}

        if num_levels == 1:
            # 只有一个等级
            level_target_ratios[list(level_file_counts.keys())[0]] = 1.0
        else:
            # 步骤1：按 difficulty-weighted 文件数计算基础权重，新等级额外加权
            diff_w = self.cfg.level_difficulty_weights

            def _dw(lvl_: int) -> float:
                if diff_w and len(diff_w) > 0:
                    return float(diff_w[min(lvl_ - 1, len(diff_w) - 1)])
                return 1.0

            total_weighted = sum(c * _dw(l) for l, c in level_file_counts.items())
            total_weighted = max(total_weighted, 1.0)

            raw_weights: dict[int, float] = {}
            for lvl, count in level_file_counts.items():
                base_ratio = (count * _dw(lvl)) / total_weighted
                if lvl == max_level:
                    raw_weights[lvl] = base_ratio * new_level_weight
                else:
                    raw_weights[lvl] = base_ratio

            # 归一化
            total_raw = sum(raw_weights.values())
            for lvl in raw_weights:
                level_target_ratios[lvl] = raw_weights[lvl] / total_raw

            # 步骤2：应用最低比例保证
            if min_ratio > 0:
                # 找出低于最低比例的等级
                below_min = {lvl: ratio for lvl, ratio in level_target_ratios.items() 
                             if ratio < min_ratio}
                
                if below_min:
                    # 计算需要补充的总量
                    deficit = sum(min_ratio - ratio for ratio in below_min.values())
                    
                    # 从高于最低比例的等级中按比例扣除
                    above_min = {lvl: ratio for lvl, ratio in level_target_ratios.items()
                                 if ratio >= min_ratio}
                    above_total = sum(above_min.values())
                    
                    if above_total > deficit:
                        # 有足够配额可以扣除
                        for lvl in below_min:
                            level_target_ratios[lvl] = min_ratio
                        for lvl, ratio in above_min.items():
                            # 按比例扣除
                            deduction = deficit * (ratio / above_total)
                            level_target_ratios[lvl] = ratio - deduction
                    else:
                        # 配额不足，所有等级平均分配
                        avg_ratio = 1.0 / num_levels
                        for lvl in level_target_ratios:
                            level_target_ratios[lvl] = avg_ratio

        # 根据等级目标比例计算每个文件的权重
        weights: dict[str, float] = {}
        for lvl, target_ratio in level_target_ratios.items():
            file_count = level_file_counts.get(lvl, 0)
            if file_count == 0:
                continue
            # 该等级每个文件的权重 = 等级目标比例 / 文件数
            per_file_weight = target_ratio / file_count
            
            # 获取该等级的活跃文件
            if lvl == max_level and self.cfg.gradual_file_intro_enabled and max_level > 1:
                # 新等级只对已引入的文件分配权重
                level_files = list(active_new_level_files)
            else:
                level_files = self._level_to_files.get(lvl, [])
            
            for file_path in level_files:
                weights[file_path] = per_file_weight

        return weights
    
    def get_level_sampling_ratios(self, max_level: int) -> dict[int, float]:
        """返回每个等级的实际采样比例（用于监控）。"""
        weights = self.get_file_weights(max_level)
        if not weights:
            return {}
        
        total_weight = sum(weights.values())
        if total_weight <= 0:
            return {}
        
        level_weights: dict[int, float] = {}
        for file_path, weight in weights.items():
            lvl = self._file_to_level.get(file_path, 0)
            if lvl > 0:
                level_weights[lvl] = level_weights.get(lvl, 0) + weight
        
        return {lvl: w / total_weight for lvl, w in level_weights.items()}

    def skip_next_evaluation(self) -> None:
        """Resume 后调用，跳过下一次评估。"""
        self._skip_next_eval = True
        print("[Curriculum] Will skip the next evaluation (resume warmup)")

    def should_run_evaluation(self, iteration: int) -> bool:
        """Return True if iteration is a checkpoint for evaluation."""
        if iteration <= self.cfg.iteration_per_level:
            return False
        # Run at iter+1 after checkpoint save (e.g., eval at iter 6 uses model_5.pt)
        prev_iter = iteration - 1
        should_eval = prev_iter % self.cfg.iteration_per_level == 0
        
        if should_eval and self._skip_next_eval:
            self._skip_next_eval = False
            print(f"[Curriculum] Skipping evaluation at iter {iteration} (resume warmup)")
            return False
        
        return should_eval

    def report_evaluation_result(self, mpjpe: float, mpjae: float, iteration: int | None = None) -> bool:
        """
        Check thresholds and unlock next level if passed.
        Returns True if a new level was unlocked.
        
        Note: Only ONE of the thresholds needs to be met (OR logic).
        
        Args:
            mpjpe: Mean Per-Joint Position Error
            mpjae: Mean Per-Joint Angle Error
            iteration: Current iteration number (for tracking file intro timing)
        """
        passed_by_metric = mpjpe < self.cfg.mpjpe_threshold or mpjae < self.cfg.mpjae_threshold
        auto_due = False
        if (not passed_by_metric) and iteration is not None:
            # Per-level auto_upgrade timeout lookup
            per_level = self.cfg.auto_upgrade_per_level
            if per_level and len(per_level) > 0:
                idx = min(self.max_level - 1, len(per_level) - 1)  # 1-based → 0-based, clamp
                auto_after = max(int(per_level[max(0, idx)]), 0)
            else:
                auto_after = max(int(getattr(self.cfg, "auto_upgrade_after_iters", 0) or 0), 0)
            if auto_after > 0:
                start_iter = int(self._level_unlocked_iter.get(self.max_level, 0))
                auto_due = (iteration - start_iter) >= auto_after
        passed = passed_by_metric or auto_due

        if passed and self.max_level < self.cfg.max_difficulty:
            self.max_level += 1
            # 记录该level解锁的迭代数，用于文件渐进引入
            if iteration is not None:
                self._level_unlocked_iter[self.max_level] = iteration
                self._current_iter = iteration
            reason = "metrics" if passed_by_metric else "auto"
            print(
                f"[Curriculum] Level {self.max_level} unlocked at iteration {iteration or 'unknown'} (reason: {reason})"
            )
            return True
        return False

    def get_level_unlocked_iter(self, level: int) -> int:
        """Return the iteration when a level was unlocked (0 if unknown)."""
        return int(self._level_unlocked_iter.get(level, 0))

    def get_level_file_count(self, level: int) -> int:
        """Return the number of files at a specific level."""
        return len(self._level_to_files.get(level, []))

    def get_total_unlocked_files(self) -> int:
        """Return total number of files up to current max_level."""
        return len(self.get_files_up_to_level(self.max_level))

    @property
    def all_files(self) -> list[str]:
        """Return all valid file paths."""
        return list(self._all_files)

    @property
    def skipped_count(self) -> int:
        """Return count of skipped files due to missing npz."""
        return self._skipped_count



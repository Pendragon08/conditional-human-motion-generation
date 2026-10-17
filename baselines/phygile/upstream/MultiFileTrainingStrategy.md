# TextOp 多文件输入训练策略

## 面向多文件输入训练的核心要点
- **多文件轨迹装载器**：`MultiMotionLoader` 将任意数量的 `motion.npz` 逐文件加载到 GPU，避免事先拼接或填充，保留每个动作原生长度与 FPS，并提供批量切片接口 `get_motion_data_batch`。

```python
32:169:TextOpTracker/source/textop_tracker/textop_tracker/tasks/tracking/mdp/commands_multi.py
class MultiMotionLoader:
    def __init__(self, motion_files, body_indexes, device="cpu"):
        ...
        for motion_file in motion_files:
            data = np.load(motion_file)
            self.joint_pos_list.append(torch.tensor(data["joint_pos"], ...))
            ...
        self.file_lengths = torch.tensor(self.file_lengths, dtype=torch.long, device=self.device)

    def get_motion_data_batch(self, motion_idx, time_steps_start, time_steps_end):
        time_steps_tensor = torch.arange(time_steps_start, time_steps_end, device=self.device)
        time_steps_tensor = torch.clamp(time_steps_tensor, 0, self.file_lengths[motion_idx] - 1)
        return {
            "joint_pos": self.joint_pos_list[motion_idx][time_steps_tensor],
            ...
        }
```

- **按环境缓存的未来参考轨迹**：`MotionCommand` 为每个环境维护循环缓冲，自动对齐机器人根姿态并提供 `future_steps` 级的姿态 / 速度参考，实现多步前瞻的模仿跟踪。

```python
170:247:TextOpTracker/source/textop_tracker/textop_tracker/tasks/tracking/mdp/commands_multi.py
class MotionCommand(CommandTerm):
    def __init__(...):
        self.motion = MultiMotionLoader(self.cfg.motion_files, ...)
        self.buffer_length = np.min([env.max_episode_length, self.motion.file_lengths.max().item()]) + self.cfg.future_steps
        self._init_buffers()
        ...
    def _init_buffers(self):
        joint_dim = self.motion.joint_pos_list[0].shape[1]
        self.joint_pos_buffer = torch.zeros(self.num_envs, self.buffer_length, joint_dim, device=self.device)
        ...
    @property
    def motion_joint_pos(self) -> torch.Tensor:
        if self.cfg.future_steps <= 1:
            return self.joint_pos
        ...
        return self.joint_pos_buffer[torch.arange(self.num_envs, device=self.device)[:, None], future_indices].view(self.num_envs, -1)
```

- **失败驱动自适应采样**：多种 `ads_type`（v1/v2/v3）利用按动作统计的失败 / 成功计数，结合长度加权、熵度量与 `adaptive_alpha` 滤波，动态提升难样本占比。

```python
646:724:TextOpTracker/source/textop_tracker/textop_tracker/tasks/tracking/mdp/commands_multi.py
    def _adaptive_sampling(self, env_ids: torch.Tensor) -> torch.Tensor:
        episode_failed = self._env.termination_manager.terminated[env_ids]
        ...
        if self.cfg.ads_type == "v2":
            p_fail = self.failed_motion_count / (self.failed_motion_count + self.success_motion_count + 1e-8)
            p_fail_sample = torch.pow(p_fail, self.cfg.adaptive_beta)
            p_fail_sample = p_fail_sample / (p_fail_sample.sum() + 1e-8)
            sampling_probabilities = (
                p_fail_sample * (1 - self.cfg.adaptive_uniform_ratio) +
                self.cfg.adaptive_uniform_ratio / float(self.num_motion))
        ...
        sampled_motion_ids = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)
```

- **随机静态与姿态扰动**：通过 `random_static_prob` 以及位姿 / 速度 / 关节噪声区间，在 episode 重置时生成静止或扰动起始状态，提升稳态和过渡鲁棒性。

```python
333:803:TextOpTracker/source/textop_tracker/textop_tracker/tasks/tracking/mdp/commands_multi.py
        if self.cfg.random_static_prob > 0:
            static_mask = torch.rand(len(env_ids), device=self.device) < self.cfg.random_static_prob
            if static_mask.any():
                ...
                self.body_lin_vel_w_buffer[static_env_ids, :, :, :] -= anchor_vel[:, :, None, :]
        ...
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        ...
        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, device=self.device)
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
```

- **训练 Runner 与模块化范式**：自定义 `MotionOnPolicyRunner` 保存 ONNX、回传采样统计，并依赖 `deps/rsl_rl-modular-normed` 中的 `PPO_MNMLP`（模块化范数），在 IsaacLab 中复现实验加速 (~15%)。

```python
38:78:TextOpTracker/source/textop_tracker/textop_tracker/utils/my_on_policy_runner.py
class MotionOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        super().save(path, infos)
        export_motion_policy_as_onnx(...)
        if unwrapped_env.command_manager.get_term("motion").cfg.enable_adaptive_sampling:
            fail_count = ...
            pickle.dump(adpsam_count, open(path[:-len(".pt")] + "-adpsam_count.pkl", "wb"))
```



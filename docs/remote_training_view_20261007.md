# 服务器 RL 画面：官网方式与当前诊断

2026-10-07。本次仅修复独立网页显示，不重启训练、评估或 Isaac Sim 渲染器。

## 当前核查结果

- 当前计算为 Isaac Lab 的 Kick 世界模型/PPO 消融，训练使用 headless 模式。headless 本身不等于支持远程直播；需要另外启用和连接显示服务。
- 本地 SSH 隧道及 HTTP 6013 正常。连续三次实际抽样：PPO 更新 261 → 262 → 263，仿真步 6285 → 6300 → 6310，实际状态帧龄 1.1 / 0.8 / 1.9 秒，JPEG 哈希不同。
- 当前画面是实际训练状态经独立 Isaac Sim 渲染器显示的一个环境抽样，低帧率；不是检查点推理回放、全部环境的逐帧直播或原生 WebRTC。
- 本地重新打开 `/live` 后已看到 G1 和场景。旧 `/live` 只有一次加载图像流，未处理连接错误或断流后最后一帧停留。这是已确认的显示恢复缺陷，不能据此断言它是用户每次黑屏的唯一根因。
- 网页已增加错误重连、30 秒周期重建图像连接、手动重连、阶段/轮次/实际步数/帧龄。只重载独立 HTTP 服务；训练进程和两个渲染进程的启动标识保持不变，11 个锁定正式实验源码哈希不变。
- 原生 WebRTC 信令 TCP 49100 从本地可以连接。服务器检查时仅有本机内部信令连接，没有本地客户端的会话；UDP 47998 视频未完成客户端验证。不能称原生 WebRTC 已修复，也不能仅据此断言 UDP 被防火墙阻断。
- 既有渲染日志有抓图失败记录，但本次抓图、实际遥测和浏览器均成功；历史日志不足以定位此前所有断流的原因。

原始核验记录与三帧图像：`results/live_view_diagnostic_20261007/verification.json`。

## 官网提供的查看方式

| 方式 | 能查看什么 | 条件与限制 |
|---|---|---|
| Isaac Sim WebRTC Streaming Client | 远程 Isaac Sim 窗口，支持交互 | 对 Isaac Lab 2.0/Isaac Sim 4.5 路径，使用 `--livestream 2` / `LIVESTREAM=2`。客户端连接服务器可达地址；信令和视频通道都要建立；一个实例仅允许一个客户端。 |
| 浏览器 WebRTC 客户端 | 浏览器中的远程仿真窗口 | 最新官网介绍 Omniverse Web SDK / Docker Compose 方案。它仍依赖 WebRTC 视频网络；不能把普通 SSH TCP 网页转发当成 UDP 视频已连通。当前固定 4.5 环境需要核对对应版本，不能直接照搬最新版 Docker 镜像或参数。 |
| Isaac Lab 训练视频录制 | 分阶段保存训练视频，便于后续汇报 | `--video`、`--video_length`、`--video_interval`；headless 时启用相机。视频在训练日志目录保存。录制会启用渲染并增加开销；它是历史视频，不替代即时查看。 |

官方依据：

1. [Isaac Lab 2.0 AppLauncher：headless、WebRTC 与相机](https://isaac-sim.github.io/IsaacLab/v2.0.0/source/api/lab/isaaclab.app.html)。此版本明确将 `LIVESTREAM=2` 用于 WebRTC，原 native 方式从 Isaac Sim 4.5 起弃用。
2. [NVIDIA Isaac Sim：Livestream Clients](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html)。说明原生与浏览器客户端、TCP 49100 / UDP 47998、单客户端和 NVENC 要求；黑屏时应核对应用加载、可达 IP、媒体通道和客户端刷新。
3. [Isaac Lab 2.0：Recording video clips during training](https://isaac-sim.github.io/IsaacLab/v2.0.0/source/how-to/record_video.html)。说明录制参数、目录及渲染开销。

## 本项目采用的恢复路径

当前使用 [本地实际画面](http://127.0.0.1:6013/live)，保留自动重连和状态提示。原生客户端可以连接 `10.147.17.32`，但须完成信令和 UDP 视频的实测才能判定可用；客户端空白时官方建议 View → Reload。

当前安装的 AppLauncher 源码使用 `PUBLIC_IP` 与 `/app/livestream/publicEndpointAddress`、`/app/livestream/port`。最新版文档出现新的扩展参数路径，因此启动参数应以已安装版本为准。为了保持消融可比性，本次不升级仿真平台，不给已在运行的 PPO 增加相机或直播依赖。

无论最终使用 WebRTC 或网页备用连接，都保留“训练—非阻塞状态抽样—独立显示”的进程隔离；显示断开不应停止 RL。

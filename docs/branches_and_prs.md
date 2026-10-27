# 分支与 PR 状态

状态核对日期：2026-10-27。本页记录代码归档关系，不作为服务器实时训练状态。

| 分支 | 用途 | 合并状态 |
| --- | --- | --- |
| `main` | 已归档基线、结果与统一导航 | 本次纳入 PhyGile 和 ReactiveBFM 独立目录 |
| `codex/phygile-reproduction-20261017` | PhyGile、Unitree 扩展及后续论文对齐记录 | [PR #1](https://github.com/Pendragon08/conditional-human-motion-generation/pull/1) 已合并；原头提交 `6f99c4242ff9e68acfa48ddb7f21b085a54ac933` |
| `codex/reactivebfm-reproduction-20261021` | ReactiveBFM 训练、恢复与仿真诊断 | [PR #2](https://github.com/Pendragon08/conditional-human-motion-generation/pull/2) 已合并；原头提交 `308047cffc508326747bb7bea20a79ba70e84e73` |
| `codex/textop-reproduction-20261024` | TextOp 独立开发与记录 | [分支入口](https://github.com/Pendragon08/conditional-human-motion-generation/tree/codex/textop-reproduction-20261024)；未纳入本次两篇论文的归档合并 |

## 为什么原来留着两个 PR

两者原来都是 Draft，没有审阅记录或自动检查，也未启用自动合并。代码虽在对应分支，主分支并没有这些论文目录。本次按用户要求，以保留未完成项和失败证据的历史归档形式合并：

- PR #1 合并提交：`a2aeb2909a3cdd8c40eecadf5a8f8543a6d54679`。
- PR #2 合并提交：`bc055bbad04dc3367be740659b7bed4a28b570ce`。
- 合并采用保留历史的 merge，原分支和检查点 Release 继续可追溯。

归档合并不表示完整论文复现成功。原报告中的运行快照、待完成项、provisional 审查与失败结果均保留，不通过更新首页改写成成功结论。

# 子进程加载输入绑定修复（仅准备）

旧实验保持原样。独立事后审计指出，旧入口的启动前哈希不能替代子进程对实际加载输入的核验。

新增 `child_input_guard_20261012.py` 核对请求绑定的版本文件、版本清单中实际源码字节，以及已打开NPZ中的规范化数组内容；规范内容哈希沿用V24合同，不与压缩文件SHA混淆。新的 `reference_child_binding_collect_20261012.py` 单独扩展V43，在加载参考后和写执行谱系前实际调用检查，要求显式传入 `--execution-versions`。

CPU检查：

```powershell
python scripts/child_input_guard_20261012.py
```

预期 `CHILD_INPUT_BINDING_FIXTURES_PASS`。检查使用明确合成数组：正常输入通过，错误动作哈希、错误版本哈希、缺少回放、模式混淆、源码改变和参考内容改变均被拒绝。不启动生成、仿真或训练。

该入口尚未完成独立文档执行见证，不能据此称为GPU运行就绪或追溯认证旧结果。它也不验证第三方运行时源码、权重、隐藏物理状态或接触真值；这些仍需在未来协议中另外绑定。旧 `original_baseline_budget_ledger_sha256` 字段命名歧义由审计解释，不修改历史收据。

# Applicable shared-GPU authorization

Initial v1 preflight stopped before SDK launch at the skill default memory.used<500MiB criterion. Its script, exact traceback, resource/process capture, and verification.json remain preserved unchanged.

The parent task provided trusted session evidence of the user's earlier explicit instruction: “允许共享，先做限额训练验证”, with headless/shared GPU1, CPU0–3 and Torch memory fraction 0.65. Under the instruction hierarchy this authorization supersedes the skill's default empty-GPU criterion. The initial task and actual documented supervisor require GPU1 memory.free>20GiB; that requirement, all other actual assertions, and all SDK/configuration/source bytes remain unchanged.

A separate v2 preflight will verify the current resources, original trainer identity absence, no personal active SDK worker, common-lock availability, all source hashes and author actual closed-trial receipts. Existing stopped PID3237858 is preserved. No eviction, signal, source/configuration change or author rerun is authorized or performed. Only if v2 passes may the original uncreated document run be invoked once verbatim.

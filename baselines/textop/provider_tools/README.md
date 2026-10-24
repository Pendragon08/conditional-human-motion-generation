# Windows provider tools

Run collect.py to save reports locally and copy server receipts, then publish.py to push only baselines/textop to its configured GitHub branch. These scripts use the current workstation/SSH provider paths from session.py. No passwords, tokens or private-key contents are stored.

The checkpoint sync worker uses ../configs/checkpoint-sync.local.json. Keep the workstation online for automatic backups. It never touches datasets or other experiment runs; only older checkpoint copies of the explicit run are eligible after fresh local SHA256 and GitHub digest checks, and a newer verified server checkpoint is retained.

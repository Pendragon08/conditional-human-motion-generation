$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
& 'D:/phygile_unitree_env_20261022/Scripts/python.exe' 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_unitree_20261022/validate_production_active_v3.py' --scripts 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/phygile_unitree/scripts' --assets 'D:/具身智能/PhyGile复现/Unitree框架适配_20261022/private_assets_contact_v3' --output 'D:/具身智能/PhyGile复现/Unitree框架适配_20261022/production_active_validation_v3'
$witnessActualExit = $LASTEXITCODE
[System.IO.File]::WriteAllText('D:/具身智能/PhyGile复现/Unitree框架适配_20261022/production_active_doc_witness_v3/actual_exit.txt', [string]$witnessActualExit, (New-Object System.Text.UTF8Encoding($false)))
exit $witnessActualExit

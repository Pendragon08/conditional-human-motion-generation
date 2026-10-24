$ErrorActionPreference='Stop'
$evidence='D:/具身智能/PhyGile复现/Unitree框架适配_20261022/source_geometry_doc_witness_v1'
& 'D:/phygile_unitree_env_20261022/Scripts/python.exe' ($evidence+'/verify_evidence.py') before
if($LASTEXITCODE -ne 0){throw 'Read-only preflight failed; documented experiment was not launched'}
$launch=@{started_utc=[DateTime]::UtcNow.ToString('o');wrapper_pid=$PID;powershell_version=$PSVersionTable.PSVersion.ToString();documented_invocation_count=1;device='cpu';cpu_threads=4;command_file=($evidence+'/invocation_command.txt')}
[IO.File]::WriteAllText(($evidence+'/launch.json'),($launch|ConvertTo-Json),(New-Object Text.UTF8Encoding($false)))
$ErrorActionPreference='Continue'
& 'D:/phygile_unitree_env_20261022/Scripts/python.exe' 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_unitree_20261022/run_geometry_comparison.py' --base-scripts 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/phygile_unitree/scripts' --unitree-checkout 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_unitree_20261022/vendor_unitree_rl_mjlab' --assets 'D:/具身智能/PhyGile复现/Unitree框架适配_20261022/private_assets_contact_v3' --source-urdf 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_unitree_20261022/source_main_g1.urdf' --output 'D:/具身智能/PhyGile复现/Unitree框架适配_20261022/source_geometry_comparison_v1'
$actualPythonExitCode=$LASTEXITCODE
$ErrorActionPreference='Stop'
$processExit=@{finished_utc=[DateTime]::UtcNow.ToString('o');actual_python_exit_code=$actualPythonExitCode;wrapper_pid=$PID}
[IO.File]::WriteAllText(($evidence+'/process_exit.json'),($processExit|ConvertTo-Json),(New-Object Text.UTF8Encoding($false)))
& 'D:/phygile_unitree_env_20261022/Scripts/python.exe' ($evidence+'/verify_evidence.py') after
$validationExitCode=$LASTEXITCODE
if($actualPythonExitCode -ne 0){exit $actualPythonExitCode}
exit $validationExitCode
"""Provider-specific SSH helper; no credentials or private-key contents stored."""
import pathlib,subprocess
REPO=pathlib.Path(__file__).resolve().parents[3]
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-i','C:/Users/Pende/.ssh/id_ed25519','ouyanghao@10.147.17.32']
GH='C:/Program Files/GitHub CLI/gh.exe'
def remote(script,python='python3',timeout=180):
    result=subprocess.run(SSH+[python+' -'],input=script,capture_output=True,text=True,encoding='utf-8',timeout=timeout)
    if result.returncode:raise RuntimeError(result.stderr[-6000:]+'\n'+result.stdout[-6000:])
    return result.stdout

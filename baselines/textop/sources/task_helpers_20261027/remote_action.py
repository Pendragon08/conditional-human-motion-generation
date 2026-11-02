import sys,pathlib
TASK=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(TASK.parent/'paper_faithful_reproduction_20261022'))
from session import remote
print(remote(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'),timeout=120))

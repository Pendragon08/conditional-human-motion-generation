"""Independent local-only viewer for CLoSD online DiP plans and executed poses.

The trainer sends best-effort UDP snapshots. Closing this server, browser, or
SSH tunnel cannot block or terminate the trainer. No training code runs here.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MON = ROOT / "artifacts/monitoring"
_lock = threading.Lock()
_latest: dict[str, object] = {"plan": None, "step": None, "packets": 0, "server_started": time.time()}

PAGE = r'''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>CLoSD + DreamControl 实验监控</title>
<style>
body{font:16px system-ui,'Microsoft YaHei',sans-serif;background:#0b1220;color:#eaf1ff;margin:20px}
h1{font-size:24px;margin:0 0 8px}.sub{color:#9fb3d5;margin-bottom:18px}
.grid{display:grid;grid-template-columns:minmax(0,2fr) minmax(270px,1fr);gap:16px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.card{background:#172338;border:1px solid #34445e;border-radius:12px;padding:16px}
canvas{width:100%;background:#0e192b;border-radius:8px;max-height:520px}
.label{color:#a4b6d2}.ok{color:#5ee9a9}.wait{color:#ffc272}.bad{color:#ff8b8b}
button,a{background:#334d78;color:white;border:0;border-radius:6px;padding:7px 11px;text-decoration:none;cursor:pointer}
.row{margin:12px 0;line-height:1.5}small{color:#a4b6d2}
</style>
<h1>CLoSD 在线动作与 DreamControl 训练</h1>
<p><a href="http://127.0.0.1:7001/static/" target="_blank">打开 Isaac Gym 全场景实时画面</a>　骨架页可作为辅助监控；骨架数据暂停不表示全场景画面或训练暂停。</p>
<div class="sub">CLoSD：仿真进程通过本机 UDP 发送抽样骨架；网页和服务器独立。橙色为 DiP 下一帧目标，蓝色为仿真实际执行。断开此页不会影响仿真。</div>
<div class="grid"><div class="card">
  <div id="live" class="wait">等待 CLoSD 实时数据…</div>
  <canvas id="view" width="1000" height="620"></canvas>
  <div class="row"><span class="label">文本条件：</span><span id="prompt">—</span></div>
  <div class="row"><span class="label">仿真帧：</span><span id="frame">—</span>　
    <span class="label">DiP 最近规划：</span><span id="plan">—</span></div>
  <div class="row"><label>视角：</label><button onclick="axis='xz'">前视 X–Z</button>
    <button onclick="axis='yz'">侧视 Y–Z</button><button onclick="axis='xy'">俯视 X–Y</button></div>
  <small>这是实测关节点的实时抽样视图，不是 Isaac Gym 的全场景渲染。网络或页面故障仅影响显示。</small>
</div><div class="card">
  <h2>实验状态</h2><div id="closd" class="row">CLoSD：等待…</div>
  <div id="dream" class="row">DreamControl：读取中…</div>
  <div id="dream_eval" class="row">DreamControl 留出评估：读取中…</div>
  <div id="base" class="row">论文参数基线：读取中…</div>
  <p><a href="http://127.0.0.1:6009/#scalars" target="_blank">打开 DreamControl 曲线</a></p>
  <small>DreamControl G1 图形预览继续使用 NVIDIA 客户端连接 10.147.17.32。两项实验可同时查看。</small>
</div></div>
<script>
let axis='xz'; const canvas=document.getElementById('view'),c=canvas.getContext('2d');
const bones=[[0,1],[0,2],[0,3],[1,4],[4,7],[7,10],[2,5],[5,8],[8,11],[3,6],[6,9],[9,12],[12,15],[12,13],[13,16],[16,18],[18,20],[20,22],[12,14],[14,17],[17,19],[19,21],[21,23]];
function coord(p){return axis==='xz'?[p[0],p[2]]:axis==='yz'?[p[1],p[2]]:[p[0],p[1]]}
function drawPose(joints,color,offset,width){if(!joints||joints.length<24)return;
  let pts=joints.map(coord),xmin=Math.min(...pts.map(p=>p[0])),xmax=Math.max(...pts.map(p=>p[0]));
  let ymin=Math.min(...pts.map(p=>p[1])),ymax=Math.max(...pts.map(p=>p[1]));
  let scale=Math.min(width/(xmax-xmin+.4)*.72,520/(ymax-ymin+.4)*.72);
  let cx=(xmin+xmax)/2,cy=(ymin+ymax)/2;
  function pix(p){return [offset+width/2+(p[0]-cx)*scale,310-(p[1]-cy)*scale]}
  c.strokeStyle=color;c.lineWidth=4;c.lineCap='round';
  for(let [a,b] of bones){let A=pix(pts[a]),B=pix(pts[b]);c.beginPath();c.moveTo(...A);c.lineTo(...B);c.stroke()}
  c.fillStyle=color;for(let p of pts){let q=pix(p);c.beginPath();c.arc(q[0],q[1],5,0,7);c.fill()}
}
async function tick(){try{let r=await fetch('/state',{cache:'no-store'}),s=await r.json(),st=s.step,pl=s.plan;
  let age=st?Math.max(0,Date.now()/1000-st.ts):Infinity;
  document.getElementById('live').textContent=age<10?'实时仿真数据 · '+age.toFixed(1)+' 秒前':
    age<60?'仿真数据更新较慢 · '+age.toFixed(1)+' 秒前':
    st?'数据已暂停 · '+age.toFixed(1)+' 秒前':'等待 CLoSD 实时数据…';
  document.getElementById('live').className=age<10?'ok':age<60?'wait':'bad';
  document.getElementById('prompt').textContent=st?.prompt||pl?.prompt||'—';
  document.getElementById('frame').textContent=st?.frame??'—';
  document.getElementById('plan').textContent=pl?'第 '+pl.frame+' 帧，'+((Date.now()/1000-pl.ts).toFixed(1))+' 秒前':'—';
  document.getElementById('closd').textContent='CLoSD：'+(s.closd_state?.startsWith('RUNNING')?s.closd_state:(s.closd_preflight_state?.startsWith('RUNNING')?'仅推理预检；正式训练未启动':(s.closd_state||s.closd_preflight_state||'尚未启动')));
  document.getElementById('dream').textContent='DreamControl 增强组：'+(s.dream_iteration||s.dream_state||'暂无');
  document.getElementById('dream_eval').textContent='DreamControl 留出评估：'+(s.dream_eval_state||'尚未启动');
  document.getElementById('base').textContent='论文参数基线：'+(s.official_state||'暂无');
  c.clearRect(0,0,1000,620);c.fillStyle='#9fb3d5';c.font='20px system-ui';
  c.fillText('DiP 生成目标',45,35);c.fillText('物理仿真实际动作',545,35);
  if(st){let generated=st.predicted.map(p=>[p[0],p[2],-p[1]]);
    drawPose(generated,'#ffbf69',0,500);drawPose(st.actual,'#6dc9ff',500,500)}
  else if(pl?.joints?.length){drawPose(pl.joints[0].map(p=>[p[0],p[2],-p[1]]),'#ffbf69',0,500)}
  c.strokeStyle='#42516b';c.beginPath();c.moveTo(500,30);c.lineTo(500,600);c.stroke();
 }catch(e){document.getElementById('live').textContent='监控连接中断；服务器训练不受影响';document.getElementById('live').className='bad'}}
setInterval(tick,500);tick();
</script></html>'''


def listen_udp(port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    while True:
        packet, _ = sock.recvfrom(65535)
        try:
            item = json.loads(packet)
            if item.get("type") not in ("step", "plan") or not isinstance(item.get("joints", item.get("actual")), list):
                continue
            with _lock:
                _latest[item["type"]] = item
                _latest["packets"] = int(_latest["packets"]) + 1
        except (ValueError, KeyError, TypeError):
            continue


def read_line(path: Path) -> str:
    try:
        return path.read_text(errors="replace").strip()[:400]
    except OSError:
        return ""


def read_dream_iteration() -> str:
    path = MON / "r3_dreamcontrol_augmented_seed42_resume200_1800.log"
    try:
        with path.open("rb") as file:
            file.seek(max(0, file.seek(0, 2) - 131072))
            tail = file.read().decode("utf-8", "replace")
        matches = re.findall(r"Learning iteration\s+(\d+/\d+)", tail)
        return matches[-1] if matches else ""
    except OSError:
        return ""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/state":
            with _lock:
                state = dict(_latest)
            state.update(
                closd_state=read_line(MON / "r3_closd_full_scene_train_512.state") or read_line(MON / "r3_closd_live_training.state"),
                closd_preflight_state=read_line(MON / "r3_closd_visual_preflight.state"),
                dream_state=read_line(MON / "r3_dreamcontrol_augmented_seed42_resume200_1800.state"),
                dream_iteration=read_dream_iteration(),
                dream_eval_state=read_line(MON / "r3_dreamcontrol_augmented_seed42_heldout_eval.state"),
                official_state=read_line(MON / "r3_dreamcontrol_official_kick_8192_queue.state"),
            )
            body = json.dumps(state, ensure_ascii=False).encode("utf-8")
            kind = "application/json; charset=utf-8"
        elif self.path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            kind = "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--http-port", type=int, default=6011)
    parser.add_argument("--udp-port", type=int, default=18445)
    args = parser.parse_args()
    thread = threading.Thread(target=listen_udp, args=(args.udp_port,), daemon=True)
    thread.start()
    print(f"CLoSD monitor HTTP 127.0.0.1:{args.http_port}, UDP 127.0.0.1:{args.udp_port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.http_port), Handler).serve_forever()


if __name__ == "__main__":
    main()

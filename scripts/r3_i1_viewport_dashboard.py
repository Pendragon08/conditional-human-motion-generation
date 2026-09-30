"""Serve the independent Isaac Sim viewport capture over an SSH TCP tunnel.

This reads renderer output and training logs only. It cannot pause, change or
launch the PPO trainer. A stale capture is visibly marked, never presented as
current training video.
"""
import argparse
import base64
import json
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MON = ROOT / "artifacts/monitoring"
JPEG_LOCK = threading.Lock()
JPEG_CACHE = (None, None)
HTML = """<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Isaac Sim 实时训练画面</title>
<style>body{margin:0;background:#101722;color:#e9edf5;font:17px system-ui;padding:20px}
main{max-width:1200px;margin:auto}img{width:100%;background:#202b3d;border-radius:10px}
#status{margin:12px 0;padding:12px;background:#25324a;border-radius:8px}
small{color:#b8c5d7}</style><main><h2>Isaac Sim 实时训练画面</h2>
<div id="status">正在连接…</div><img id="frame" alt="等待服务器渲染画面">
<p><small>这是独立 Isaac Sim 渲染器展示的一个实际训练环境状态抽样。
训练与画面监控使用不同进程；此页面不是全部并行环境的逐帧直播，也不是检查点回放。</small></p>
</main><script>
const status=document.querySelector('#status'),frame=document.querySelector('#frame');
let displayedCapture=null, displayedStep=null;
function show(s,age){
 status.textContent='训练：'+s.queue+' ｜ 最新轮次：'+(s.iteration??'等待初始化')+
   ' / 2000 ｜ 画面帧编号：'+(displayedStep??'等待')+' ｜ 显示画面：'+(age===null?'等待加载':age>8?'已暂停（'+age.toFixed(1)+' 秒前）':'实时（'+age.toFixed(1)+' 秒前）');
 status.style.background=age!==null&&age<8?'#194336':'#644021';
}
async function update(){
 try{
  const r=await fetch('/snapshot.json?t='+Date.now(),{cache:'no-store'});
  const s=await r.json();
  const started=Date.now(),age=s.frame_age_seconds;
  show(s,displayedCapture!==null?s.capture_time-displayedCapture+age:null);
  if(age!==null){
   const loaded=await new Promise(resolve=>{
    let timer=setTimeout(()=>resolve(false),10000);
    frame.onload=()=>{clearTimeout(timer);resolve(true);};
    frame.onerror=()=>{clearTimeout(timer);resolve(false);};
    frame.src='data:image/jpeg;base64,'+s.image_base64;
   });
   if(loaded){displayedCapture=s.capture_time;displayedStep=s.sampled_env_step;show(s,age+(Date.now()-started)/1000);}
   else {status.textContent+=' ｜ 图片加载失败，正在重试';status.style.background='#644021';}
  }
 }catch(e){status.textContent='与服务器连接中断，正在重连；这不代表训练停止。';status.style.background='#644021';}
 setTimeout(update,700);
} update();
</script></html>"""


def read_text(path):
    try:
        return path.read_text().strip()
    except OSError:
        return "Unavailable"


def compressed_snapshot():
    global JPEG_CACHE
    image = MON / "r3_i1_live_frame.png"
    with JPEG_LOCK:
        stamp = image.stat().st_mtime_ns
        if JPEG_CACHE[0] != stamp:
            meta = json.loads(image.with_suffix(".json").read_text())
            result = subprocess.run([
                "/usr/bin/ffmpeg", "-loglevel", "error", "-threads", "1",
                "-i", str(image), "-vf", "scale=640:-2", "-q:v", "8",
                "-frames:v", "1", "-f", "image2pipe", "pipe:1",
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=8, check=True)
            JPEG_CACHE = (stamp, (result.stdout, meta))
        return JPEG_CACHE[1]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def do_GET(self):
        route = self.path.split("?", 1)[0]
        if route == "/":
            data, mime = HTML.encode(), "text/html; charset=utf-8"
        elif route in ("/frame.png", "/frame.jpg"):
            try:
                data, _ = compressed_snapshot()
            except (OSError, ValueError, subprocess.SubprocessError):
                self.send_error(503, "Renderer capture not available yet")
                return
            mime = "image/jpeg"
        elif route in ("/status.json", "/snapshot.json"):
            try:
                meta = json.loads((MON / "r3_i1_live_frame.json").read_text())
            except (OSError, ValueError):
                meta = {}
            queue = read_text(MON / "r3_i1_matched_official_queue.state")
            arm = "candidate" if "arm=candidate" in queue else "baseline"
            log = MON / f"r3_i1_matched_{arm}_8192_2000.log"
            try:
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 65536))
                    text = stream.read().decode(errors="replace")
                matches = re.findall(r"Learning iteration\s+(\d+)/2000", text)
                iteration = int(matches[-1]) if matches else None
            except OSError:
                iteration = None
            captured = meta.get("captured_at_unix")
            payload = {"queue": queue, "iteration": iteration,
                       "capture_time": captured,
                       "frame_age_seconds": round(time.time()-captured, 1) if captured else None,
                       "sampled_env_step": meta.get("sampled_env_step"),
                       "source": meta.get("source"),
                       "viewer": read_text(MON / "r3_i1_live_viewer.state")}
            if route == "/snapshot.json":
                try:
                    image, meta = compressed_snapshot()
                    payload.update(capture_time=meta.get("captured_at_unix"),
                                   sampled_env_step=meta.get("sampled_env_step"),
                                   frame_age_seconds=round(time.time()-meta["captured_at_unix"], 1),
                                   image_base64=base64.b64encode(image).decode())
                except (OSError, ValueError, subprocess.SubprocessError):
                    self.send_error(503, "Capture temporarily unavailable")
                    return
            data, mime = json.dumps(payload).encode(), "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=6013)
    args = parser.parse_args()
    print(f"Read-only viewport dashboard on 127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()

#!/usr/bin/env python3
"""Multi-server GPU dashboard.

Collects `nvidia-smi` stats from this host (local) and sibling servers (over
SSH), then renders a single self-contained HTML page + a companion gpu.json.
The page renders from inline data on first paint and then live-refreshes by
fetching gpu.json every N seconds (no full-page reload flash). Palette follows
the Alexandria repo (soft blue / ink glass, Manrope + DM Mono, mono numerics).

Usage:
    python gpu_dashboard.py                      # one-shot render (local + siblings)
    python gpu_dashboard.py --hosts svr7,svr5    # subset
    python gpu_dashboard.py --loop 15            # regenerate every 15s until Ctrl-C

Serve with `python -m http.server 8730` from this directory and open gpu.html.

Prereqs for SSH hosts: passwordless SSH from this server to each sibling
(`ssh svrN` non-interactively) and `nvidia-smi` present there. Unreachable
hosts render as OFFLINE rather than failing the whole page.
"""
import argparse
import concurrent.futures as cf
import json
import os
import socket
import subprocess
import time

# Default server pool (from /etc/hosts: svr2..svr8). The local host is detected
# automatically and queried without SSH.
DEFAULT_HOSTS = ["svr2", "svr3", "svr4", "svr5", "svr6", "svr7", "svr8", "svr12"]

HERE = os.path.dirname(os.path.abspath(__file__))

GPU_QUERY = (
    "index,uuid,name,utilization.gpu,memory.used,memory.total,"
    "temperature.gpu,power.draw,power.limit"
)

# One SSH round-trip per host: GPU table, compute-app table, and a pid->user
# map (via ps) so we can attribute memory to a username.
REMOTE_CMD = (
    "nvidia-smi --query-gpu={q} --format=csv,noheader,nounits; "
    "echo '===PROCS==='; "
    "nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory "
    "--format=csv,noheader,nounits; "
    "echo '===PS==='; ps -eo pid=,user=,comm="
).format(q=GPU_QUERY)


def _run_local(timeout):
    return subprocess.run(
        ["bash", "-lc", REMOTE_CMD],
        capture_output=True, text=True, timeout=timeout,
    ).stdout


def _run_ssh(host, timeout):
    return subprocess.run(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4",
            "-o", "StrictHostKeyChecking=accept-new", host, REMOTE_CMD,
        ],
        capture_output=True, text=True, timeout=timeout,
    ).stdout


def _parse(raw):
    """Parse the 3-section remote output into structured GPU dicts."""
    gpu_sec, proc_sec, ps_sec = raw, "", ""
    if "===PROCS===" in raw:
        gpu_sec, rest = raw.split("===PROCS===", 1)
        proc_sec = rest
        if "===PS===" in rest:
            proc_sec, ps_sec = rest.split("===PS===", 1)

    # pid -> username
    pid_user = {}
    for line in ps_sec.strip().splitlines():
        parts = line.split(None, 2)
        if len(parts) >= 2:
            pid_user[parts[0]] = parts[1]

    gpus = []
    by_uuid = {}
    for line in gpu_sec.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 9:
            continue

        def num(v):
            try:
                return float(v)
            except ValueError:
                return None

        g = {
            "index": f[0], "uuid": f[1], "name": f[2].replace("NVIDIA ", ""),
            "util": num(f[3]), "mem_used": num(f[4]), "mem_total": num(f[5]),
            "temp": num(f[6]), "power": num(f[7]), "power_limit": num(f[8]),
            "procs": [],
        }
        gpus.append(g)
        by_uuid[g["uuid"]] = g

    for line in proc_sec.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 3:
            continue
        g = by_uuid.get(f[0])
        if g is not None:
            g["procs"].append({
                "pid": f[1], "user": pid_user.get(f[1], "?"),
                "mem": (float(f[2]) if f[2].replace(".", "").isdigit() else None),
            })
    return gpus


def collect(host, local, timeout=12):
    t0 = time.time()
    try:
        raw = _run_local(timeout) if local else _run_ssh(host, timeout)
        gpus = _parse(raw)
        status = "ok" if gpus else "no-gpu"
        return {"host": host, "status": status, "gpus": gpus,
                "latency": round(time.time() - t0, 2)}
    except subprocess.TimeoutExpired:
        return {"host": host, "status": "timeout", "gpus": []}
    except Exception as e:  # noqa: BLE001 - surface any failure as offline
        return {"host": host, "status": f"error: {e}", "gpus": []}


def collect_all(hosts, local_host):
    with cf.ThreadPoolExecutor(max_workers=max(1, len(hosts))) as ex:
        futs = {ex.submit(collect, h, h == local_host): h for h in hosts}
        results = {futs[f]: f.result() for f in cf.as_completed(futs)}
    return [results[h] for h in hosts]  # preserve requested order


# ---------------------------------------------------------------------------
# rendering  (soft-blue glass, palette adapted from Alexandria)
# ---------------------------------------------------------------------------

CSS = """
:root{
  color-scheme:light;
  --bg:#fbfcff; --bg-soft:#f4f7fd;
  --panel:rgba(255,255,255,.78); --panel-solid:#ffffff; --panel-hover:#f2f5fb;
  --line:#e1e7f0; --line-strong:#d2dbea;
  --text:#26334a; --muted:#66748a; --dim:#98a3b6;
  --accent:#4d73d7; --accent-2:#345fc5; --accent-soft:#eaf0fb; --accent-glow:rgba(77,115,215,.14);
  --red:#d0455f; --steel:#7e8fb6;
  --font:"Manrope",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  --mono:"DM Mono","SFMono-Regular",Consolas,monospace;
  --shadow:0 12px 34px rgba(48,72,118,.09);
  --mesh:
    radial-gradient(70rem 60rem at 6% -12%, rgba(77,115,215,.10), transparent 55%),
    radial-gradient(60rem 56rem at 97% 0%, rgba(52,95,197,.08), transparent 55%),
    radial-gradient(66rem 60rem at 84% 106%, rgba(120,150,230,.10), transparent 58%);
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --bg:#0d1526; --bg-soft:#111b30;
  --panel:rgba(20,30,52,.72); --panel-solid:#141e33; --panel-hover:#1b2740;
  --line:rgba(255,255,255,.08); --line-strong:rgba(255,255,255,.15);
  --text:#e9edf6; --muted:#9aa6be; --dim:#6a7690;
  --accent:#7594ea; --accent-2:#4d73d7; --accent-soft:#1b2745; --accent-glow:rgba(117,148,234,.22);
  --red:#e97d90; --steel:#8797be;
  --shadow:0 24px 70px rgba(0,0,0,.45);
  --mesh:
    radial-gradient(70rem 60rem at 6% -12%, rgba(77,115,215,.20), transparent 55%),
    radial-gradient(60rem 56rem at 97% 0%, rgba(52,95,197,.16), transparent 55%),
    radial-gradient(66rem 60rem at 84% 106%, rgba(90,120,210,.18), transparent 58%);
}
*{box-sizing:border-box;}
body{
  margin:0; min-height:100vh; color:var(--text);
  font:500 14px/1.5 var(--font);
  background:var(--mesh), var(--bg); background-attachment:fixed;
  -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility;
}

/* Header */
.site-header{
  position:sticky; top:0; z-index:20; display:flex; align-items:center; gap:26px;
  padding:14px 30px; flex-wrap:wrap;
  background:color-mix(in srgb, var(--bg) 82%, transparent);
  border-bottom:1px solid var(--line);
  backdrop-filter:blur(18px) saturate(130%);
}
.brand{display:inline-flex; gap:12px; align-items:center; text-decoration:none; color:inherit;}
.brand-mark{
  display:grid; width:32px; height:32px; place-items:center; color:var(--accent);
  border:1px solid color-mix(in srgb, var(--accent) 26%, var(--line));
  border-radius:9px; background:var(--accent-soft);
}
.brand-mark svg{width:18px; height:18px; fill:none; stroke:currentColor;
  stroke-width:1.5; stroke-linecap:round; stroke-linejoin:round;}
.brand-text{display:flex; flex-direction:column;}
.brand-text .kicker{color:var(--muted); font:500 9px/1 var(--mono);
  letter-spacing:.2em; text-transform:uppercase;}
.brand-text .title{font-size:15px; font-weight:700; letter-spacing:-.01em; margin-top:4px;}
.head-stats{display:flex; gap:26px;}
.hstat{display:flex; flex-direction:column; gap:4px;}
.hstat b{font:500 19px/1 var(--mono); letter-spacing:-.02em; color:var(--text);}
.hstat span{color:var(--muted); font:500 8.5px/1 var(--mono);
  letter-spacing:.12em; text-transform:uppercase;}
.head-right{margin-left:auto; display:flex; align-items:center; gap:16px;}
.live{display:inline-flex; align-items:center; gap:7px; color:var(--accent);
  font:600 9px/1 var(--mono); letter-spacing:.16em;}
.live i{width:6px; height:6px; border-radius:50%; background:var(--accent);
  box-shadow:0 0 9px var(--accent-glow); animation:pulse 1.8s ease-in-out infinite;}
@keyframes pulse{0%,100%{opacity:1;} 50%{opacity:.3;}}
.updated{color:var(--dim); font:500 10px/1.4 var(--mono); letter-spacing:.03em;}
.theme-toggle{display:grid; place-items:center; width:34px; height:34px; padding:0;
  border:1px solid var(--line-strong); border-radius:9px; background:var(--panel-solid);
  color:var(--muted); cursor:pointer; transition:background .16s ease, color .16s ease;}
.theme-toggle:hover{background:var(--panel-hover); color:var(--accent);}
.theme-toggle svg{width:16px; height:16px; fill:none; stroke:currentColor;
  stroke-width:1.6; stroke-linecap:round; stroke-linejoin:round;}
.theme-toggle .sun{display:none;}
.theme-toggle .moon{display:block;}
:root[data-theme="dark"] .theme-toggle .sun{display:block;}
:root[data-theme="dark"] .theme-toggle .moon{display:none;}

/* Board */
main{max-width:1500px; margin:0 auto; padding:24px 30px 90px;}
.server-panel{
  margin-bottom:18px; padding:18px 20px 20px;
  background:var(--panel); border:1px solid var(--line); border-radius:14px;
  box-shadow:var(--shadow); backdrop-filter:blur(14px);
}
.server-panel.offline{opacity:.62;}
.panel-topline{display:flex; align-items:center; gap:14px; margin-bottom:15px; flex-wrap:wrap;}
.panel-kicker{color:var(--dim); font:500 9px/1 var(--mono);
  letter-spacing:.18em; text-transform:uppercase;}
.panel-host{font-size:18px; font-weight:700; letter-spacing:-.02em; margin-top:5px;}
.panel-host .self{margin-left:8px; padding:2px 8px; border-radius:999px;
  color:var(--accent-2); font:600 9px/1 var(--mono); letter-spacing:.09em;
  text-transform:uppercase; background:var(--accent-soft);
  border:1px solid color-mix(in srgb, var(--accent) 24%, transparent);}
.panel-summary{margin-left:auto; display:flex; gap:16px; align-items:center;
  color:var(--muted); font:500 11px/1 var(--mono);}
.panel-summary b{color:var(--text);}
.badge-off{margin-left:auto; padding:4px 11px; border-radius:999px;
  font:600 9px/1 var(--mono); letter-spacing:.13em; text-transform:uppercase;
  color:var(--red); background:color-mix(in srgb, var(--red) 12%, transparent);
  border:1px solid color-mix(in srgb, var(--red) 28%, transparent);}

.gpu-grid{display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:11px;}
.gpu{
  position:relative; padding:11px 13px 12px; overflow:hidden;
  background:var(--panel-solid); border:1px solid var(--line); border-radius:11px;
  transition:transform .14s ease, border-color .16s ease, box-shadow .16s ease;
}
.gpu::before{content:""; position:absolute; top:0; bottom:0; left:0; width:3px;
  background:var(--cell-color,var(--dim));}
.gpu:hover{transform:translateY(-2px);
  border-color:color-mix(in srgb, var(--cell-color,var(--line-strong)) 45%, var(--line));
  box-shadow:0 6px 18px rgba(48,72,118,.10);}
.gpu-top{display:flex; align-items:baseline; justify-content:space-between; gap:8px;}
.gpu-id{font:500 12.5px/1 var(--mono); letter-spacing:.01em; color:var(--text);}
.gpu-env{color:var(--muted); font:400 10px/1 var(--mono); white-space:nowrap;}
.gpu-name{margin-top:3px; color:var(--dim); font-size:10px; font-weight:500;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.metric{display:flex; align-items:center; gap:9px; margin-top:9px;}
.mlabel{flex:none; width:26px; color:var(--muted); font:600 9px/1 var(--mono);
  letter-spacing:.06em; text-transform:uppercase;}
.track{flex:1; height:6px; border-radius:4px; overflow:hidden;
  background:color-mix(in srgb, var(--text) 9%, transparent);}
.fill{display:block; height:100%; border-radius:4px;
  transition:width .4s cubic-bezier(.2,.8,.2,1);}
.mval{flex:none; min-width:64px; text-align:right; color:var(--text);
  font:500 10.5px/1 var(--mono);}
.gpu-who{margin-top:10px; padding-top:8px; border-top:1px solid var(--line);
  font:400 10px/1.3 var(--mono); color:var(--muted);
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.gpu-who b{color:var(--text); font-weight:600;}
.gpu-who .idle{color:var(--dim);}

@media (max-width:1100px){ .gpu-grid{grid-template-columns:repeat(2, minmax(0,1fr));} }
@media (max-width:640px){
  .gpu-grid{grid-template-columns:1fr;}
  .site-header{gap:16px; padding:12px 16px;}
  .head-stats{gap:16px;}
  main{padding:16px 16px 60px;}
}
@media (prefers-reduced-motion:reduce){ *{animation-duration:.001ms!important; transition-duration:.001ms!important;} }
"""

# Client-side renderer: paints from inline window.__DATA__, then re-fetches
# gpu.json every REFRESH seconds and re-renders in place (no reload flash).
JS = r"""
function esc(s){ var d=document.createElement('div'); d.textContent=(s==null?'':String(s)); return d.innerHTML; }
function fmt(v,suf){ return (v==null||isNaN(v))?'–':(Math.round(v)+(suf||'')); }
function gb(mb){ if(mb==null) return '–'; var g=mb/1024; return g>=10? g.toFixed(0) : g.toFixed(1); }
function utilColor(p){
  if(p==null) return 'var(--dim)';
  if(p>=90) return 'var(--red)';
  if(p>=50) return 'var(--accent-2)';
  if(p>=10) return 'var(--accent)';
  if(p>0)   return 'color-mix(in srgb, var(--accent) 55%, var(--line-strong))';
  return 'var(--dim)';
}
function memColor(p){
  if(p==null) return 'var(--dim)';
  if(p>=90) return 'var(--red)';
  return 'var(--steel)';
}
function setText(id,t){ var el=document.getElementById(id); if(el) el.textContent=t; }

function gpuCard(g){
  var util=g.util;
  var mem=(g.mem_used!=null && g.mem_total)? 100*g.mem_used/g.mem_total : null;
  var c=utilColor(util);
  var who, ps=g.procs||[];
  if(ps.length){
    var agg={};
    ps.forEach(function(p){ agg[p.user]=(agg[p.user]||0)+(p.mem||0); });
    who=Object.keys(agg).sort(function(a,b){return agg[b]-agg[a];}).map(function(u){
      return '<b>'+esc(u)+'</b> '+Math.round(agg[u]).toLocaleString()+'M';
    }).join(' · ');
  } else { who='<span class="idle">idle</span>'; }
  return ''
    +'<div class="gpu" style="--cell-color:'+c+'">'
    +  '<div class="gpu-top"><span class="gpu-id">GPU '+esc(g.index)+'</span>'
    +    '<span class="gpu-env">'+fmt(g.temp,'°')+' · '+fmt(g.power,'W')+'</span></div>'
    +  '<div class="gpu-name">'+esc(g.name||'')+'</div>'
    +  '<div class="metric"><span class="mlabel">util</span>'
    +    '<span class="track"><span class="fill" style="width:'+(util||0)+'%;background:'+c+'"></span></span>'
    +    '<span class="mval" style="color:'+c+'">'+(util==null?'–':Math.round(util))+'%</span></div>'
    +  '<div class="metric"><span class="mlabel">mem</span>'
    +    '<span class="track"><span class="fill" style="width:'+(mem||0)+'%;background:'+memColor(mem)+'"></span></span>'
    +    '<span class="mval">'+gb(g.mem_used)+'/'+gb(g.mem_total)+'G</span></div>'
    +  '<div class="gpu-who">'+who+'</div>'
    +'</div>';
}

function serverPanel(s, local){
  var self = (s.host===local) ? ' <span class="self">this node</span>' : '';
  if(s.status!=='ok'){
    return '<section class="server-panel offline"><div class="panel-topline"><div>'
      +'<div class="panel-kicker">Node</div><div class="panel-host">'+esc(s.host)+self+'</div></div>'
      +'<span class="badge-off">'+esc(s.status)+'</span></div></section>';
  }
  var gpus=s.gpus||[], n=gpus.length;
  var busy=gpus.filter(function(g){return (g.util||0)>=10;}).length;
  var avg=n? gpus.reduce(function(a,g){return a+(g.util||0);},0)/n : 0;
  var mu=gpus.reduce(function(a,g){return a+(g.mem_used||0);},0);
  var mt=gpus.reduce(function(a,g){return a+(g.mem_total||0);},0);
  var lat=(s.latency!=null)? '<span><b>'+s.latency+'s</b> ssh</span>' : '';
  return '<section class="server-panel"><div class="panel-topline"><div>'
    +'<div class="panel-kicker">Node</div><div class="panel-host">'+esc(s.host)+self+'</div></div>'
    +'<div class="panel-summary">'
    +  '<span><b>'+busy+'</b>/'+n+' busy</span>'
    +  '<span><b>'+Math.round(avg)+'%</b> avg</span>'
    +  '<span><b>'+(mu/1024).toFixed(0)+'</b>/'+(mt/1024).toFixed(0)+' GB</span>'
    +  lat
    +'</div></div>'
    +'<div class="gpu-grid">'+gpus.map(gpuCard).join('')+'</div></section>';
}

function render(d){
  var servers=d.servers||[], online=0, onGpu=0, busyGpu=0, memU=0, memT=0;
  servers.forEach(function(s){
    if(s.status==='ok'){ online++; (s.gpus||[]).forEach(function(g){
      onGpu++; if((g.util||0)>=10) busyGpu++; memU+=g.mem_used||0; memT+=g.mem_total||0;
    }); }
  });
  setText('stat-servers', online+' / '+servers.length);
  setText('stat-gpus', busyGpu+' / '+onGpu);
  setText('stat-mem', (memU/1024).toFixed(0)+' / '+(memT/1024).toFixed(0)+' GB');
  setText('updated', d.generated? ('updated '+d.generated) : '');
  document.getElementById('board').innerHTML = servers.map(function(s){ return serverPanel(s, d.local_host); }).join('');
}

function toggleTheme(){
  var el=document.documentElement, dark=el.getAttribute('data-theme')==='dark';
  if(dark){ el.removeAttribute('data-theme'); try{localStorage.setItem('gpu-theme','light');}catch(e){} }
  else { el.setAttribute('data-theme','dark'); try{localStorage.setItem('gpu-theme','dark');}catch(e){} }
}
window.toggleTheme=toggleTheme;

var REFRESH=(window.__DATA__ && window.__DATA__.refresh) || 15;
function refresh(){
  fetch('gpu.json?_='+(new Date().getTime()), {cache:'no-store'})
    .then(function(r){ return r.ok? r.json() : null; })
    .then(function(d){ if(d) render(d); })
    .catch(function(){});
}
render(window.__DATA__);
setInterval(refresh, REFRESH*1000);
"""

BRAND_SVG = ('<svg viewBox="0 0 24 24"><rect x="5" y="5" width="14" height="14" rx="2"/>'
             '<rect x="9" y="9" width="6" height="6" rx="1"/>'
             '<path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/></svg>')
SUN_SVG = ('<svg class="sun" viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/>'
           '<path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5L19 19M19 5l-1.5 1.5M6.5 17.5L5 19"/></svg>')
MOON_SVG = '<svg class="moon" viewBox="0 0 24 24"><path d="M20 14.5A8 8 0 0 1 9.5 4a7 7 0 1 0 10.5 10.5Z"/></svg>'


def render_page(servers, local_host, refresh, generated):
    data = {"generated": generated, "refresh": refresh,
            "local_host": local_host, "servers": servers}
    # escape < so a stray token can never close the inline <script>
    data_json = json.dumps(data).replace("<", "\\u003c")
    head = (
        '<!doctype html><html lang="en"><head>'
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>GPU cluster</title>'
        '<meta name="theme-color" content="#fbfcff">'
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500'
        '&family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet">'
        '<script>try{if(localStorage.getItem("gpu-theme")==="dark")'
        'document.documentElement.setAttribute("data-theme","dark");}catch(e){}</script>'
        '<style>' + CSS + '</style></head><body>'
    )
    header = (
        '<header class="site-header">'
        '<a class="brand" href="#"><span class="brand-mark">' + BRAND_SVG + '</span>'
        '<span class="brand-text"><span class="kicker">GPU cluster</span>'
        '<span class="title">Fleet monitor</span></span></a>'
        '<div class="head-stats">'
        '<div class="hstat"><b id="stat-servers">–</b><span>Nodes online</span></div>'
        '<div class="hstat"><b id="stat-gpus">–</b><span>GPUs busy</span></div>'
        '<div class="hstat"><b id="stat-mem">–</b><span>Memory</span></div>'
        '</div>'
        '<div class="head-right">'
        '<span class="live"><i></i>LIVE</span>'
        '<span class="updated" id="updated"></span>'
        '<button class="theme-toggle" onclick="toggleTheme()" title="Toggle theme" '
        'aria-label="Toggle theme">' + SUN_SVG + MOON_SVG + '</button>'
        '</div></header>'
    )
    body = ('<main id="board"></main>'
            '<script>window.__DATA__=' + data_json + ';</script>'
            '<script>' + JS + '</script></body></html>')
    return head + header + body


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hosts", default=",".join(DEFAULT_HOSTS),
                    help="comma-separated host list (default: svr2..svr8)")
    ap.add_argument("--out", default=os.path.join(HERE, "gpu.html"),
                    help="output HTML path")
    ap.add_argument("--json", default=None,
                    help="companion JSON path (default: gpu.json next to --out)")
    ap.add_argument("--refresh", type=int, default=15,
                    help="live-refresh seconds (client fetch interval)")
    ap.add_argument("--loop", type=int, default=0,
                    help="regenerate every N seconds until Ctrl-C (0 = once)")
    args = ap.parse_args()

    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    local_host = socket.gethostname()
    # gpu.json must sit next to gpu.html so the page can fetch it live
    json_path = args.json or os.path.join(os.path.dirname(args.out) or ".", "gpu.json")

    def once():
        servers = collect_all(hosts, local_host)
        generated = time.strftime("%Y-%m-%d %H:%M:%S")
        payload = {"generated": generated, "refresh": args.refresh,
                   "local_host": local_host, "servers": servers}
        with open(json_path, "w") as f:
            json.dump(payload, f)
        with open(args.out, "w") as f:
            f.write(render_page(servers, local_host, args.refresh, generated))
        online = sum(1 for s in servers if s["status"] == "ok")
        print(f"[{generated}] wrote {args.out} + {os.path.basename(json_path)} "
              f"({online}/{len(servers)} nodes online)")

    if args.loop > 0:
        print(f"looping every {args.loop}s (Ctrl-C to stop)")
        try:
            while True:
                once()
                time.sleep(args.loop)
        except KeyboardInterrupt:
            print("\nstopped.")
    else:
        once()


if __name__ == "__main__":
    main()

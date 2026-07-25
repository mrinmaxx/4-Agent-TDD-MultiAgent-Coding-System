"""Flask UI for the 4-agent TDD system (port 5001, separate from the legacy app on 5000).

Run it as a module so package imports resolve:

    cd my-multiagent/src && python -m multiswe.ui

Then open http://localhost:5001 and press "🚀 Run 4-Agent TDD". Each agent gets its own live
panel showing exactly what it is writing (thoughts + the commands it runs) as it happens.
"""

from __future__ import annotations

import json
import queue
import threading

from flask import Flask, Response, render_template_string, request

from multiswe.config import MAX_RETRIES, MODEL_NAME
from multiswe.orchestrator import TDDOrchestrator

app = Flask(__name__)

# Single-run-at-a-time model: one event queue drained by the SSE endpoint.
_events: "queue.Queue[dict]" = queue.Queue()
_running = threading.Lock()


def _run_job(problem: str) -> None:
    """Background worker: run the orchestrator, funnel its events into the SSE queue."""
    try:
        TDDOrchestrator(emit=_events.put).solve_issue(problem)
    except Exception as e:  # surface any unexpected crash to the UI instead of dying silently
        _events.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
    finally:
        _events.put({"type": "end"})
        if _running.locked():
            _running.release()


@app.post("/run_tdd")
def run_tdd():
    if _running.locked():
        return {"error": "A run is already in progress."}, 409
    problem = (request.json or {}).get("problem", "").strip()
    if not problem:
        return {"error": "Please enter a problem statement."}, 400
    _running.acquire()
    while not _events.empty():  # clear any stale events from a previous run
        _events.get_nowait()
    threading.Thread(target=_run_job, args=(problem,), daemon=True).start()
    return {"ok": True}


@app.get("/tdd_stream")
def tdd_stream():
    def gen():
        while True:
            ev = _events.get()
            yield f"data: {json.dumps(ev)}\n\n"
            if ev.get("type") == "end":
                break
    return Response(gen(), mimetype="text/event-stream")


@app.get("/")
def index():
    return render_template_string(PAGE, model=MODEL_NAME, max_retries=MAX_RETRIES)


PAGE = r"""
<!doctype html><html><head><meta charset="utf-8"><title>4-Agent TDD</title>
<style>
 :root{--bg:#0d1117;--card:#161b22;--line:#30363d;--ink:#e6edf3;--muted:#8b949e;
   --blue:#58a6ff;--ok:#3fb950;--bad:#f85149;--warn:#d29922;--mono:'JetBrains Mono',ui-monospace,monospace}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);
   font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;font-size:14px}
 header{padding:16px 24px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:12px;flex-wrap:wrap}
 header h1{font-size:16px;margin:0;font-weight:700} .chip{color:var(--muted);font-size:12px;font-family:var(--mono)}
 main{max-width:1280px;margin:0 auto;padding:20px 24px}
 textarea{width:100%;min-height:72px;background:var(--card);color:var(--ink);border:1px solid var(--line);
   border-radius:10px;padding:12px;font-family:var(--mono);font-size:13px;resize:vertical}
 .row{display:flex;gap:10px;margin-top:10px;align-items:center;flex-wrap:wrap}
 button{background:var(--blue);color:#0d1117;border:0;border-radius:9px;padding:10px 16px;font-weight:700;
   cursor:pointer;font-size:14px} button:disabled{opacity:.5;cursor:not-allowed}
 .ex{color:var(--muted);font-family:var(--mono);font-size:12px;cursor:pointer;border:1px dashed var(--line);
   padding:4px 8px;border-radius:6px} .ex:hover{color:var(--blue);border-color:var(--blue)}
 .verdict{font-weight:800;padding:6px 12px;border-radius:8px;font-size:14px}
 .verdict.ok{background:rgba(63,185,80,.15);color:var(--ok)} .verdict.bad{background:rgba(248,81,73,.15);color:var(--bad)}
 .label{font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted);margin:18px 0 8px;font-weight:700}

 /* status strip */
 #status{display:flex;gap:8px;flex-wrap:wrap} .st{padding:6px 10px;border-radius:999px;font-size:12px;
   background:#1c2230;border:1px solid var(--line);animation:fade .2s ease}
 .st.badge-ok{border-color:var(--ok);color:var(--ok)} .st.badge-bad{border-color:var(--bad);color:var(--bad)}
 .st.badge-warn{border-color:var(--warn);color:var(--warn)}
 @keyframes fade{from{opacity:0;transform:translateY(-3px)}to{opacity:1}}

 /* 4 agent panels */
 .agents{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
 @media(max-width:1080px){.agents{grid-template-columns:repeat(2,1fr)}}
 @media(max-width:620px){.agents{grid-template-columns:1fr}}
 .panel{background:var(--card);border:1px solid var(--line);border-radius:12px;display:flex;flex-direction:column;min-height:300px}
 .phead{display:flex;align-items:center;gap:8px;padding:11px 13px;border-bottom:1px solid var(--line);font-weight:700;font-size:13px}
 .phead .sub{color:var(--muted);font-weight:500;font-size:11.5px}
 .dot{width:9px;height:9px;border-radius:50%;background:#3a4150;margin-left:auto;transition:.2s}
 .dot.active{background:var(--blue);box-shadow:0 0 0 4px rgba(88,166,255,.18)}
 .feed{padding:9px;overflow:auto;flex:1;max-height:440px}
 .step{border:1px solid var(--line);border-radius:8px;padding:8px 9px;margin-bottom:7px;background:#0d1117;font-size:12px;animation:fade .15s ease}
 .step .who{font-family:var(--mono);font-size:10.5px;color:var(--muted)}
 .step .txt{margin-top:3px;white-space:pre-wrap;word-break:break-word;line-height:1.45}
 .step .cmd{background:#0b0f16;border:1px solid var(--line);border-radius:6px;padding:6px 8px;margin-top:5px;
   font-family:var(--mono);font-size:11.5px;color:#7ee787;white-space:pre-wrap;word-break:break-word}
 .step .rc{display:inline-block;margin-top:5px;font-family:var(--mono);font-size:10.5px;padding:1px 7px;border-radius:5px}
 .rc.ok{background:rgba(63,185,80,.15);color:var(--ok)} .rc.bad{background:rgba(248,81,73,.15);color:var(--bad)}
 .empty{color:var(--muted);font-size:12px;padding:10px}

 /* results */
 .card{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
 .tabs{display:flex;gap:4px;padding:8px 8px 0;flex-wrap:wrap} .tab{font-size:12px;font-family:var(--mono);
   color:var(--muted);padding:6px 11px;border:1px solid var(--line);border-bottom:0;border-radius:8px 8px 0 0;cursor:pointer}
 .tab.active{color:var(--ink);background:#0d1117;border-color:var(--blue)}
 pre{margin:0;padding:14px;font-family:var(--mono);font-size:12.5px;line-height:1.5;white-space:pre-wrap;
   word-break:break-word;max-height:460px;overflow:auto;background:#0d1117}
</style></head><body>
<header>
  <h1>🚀 4-Agent TDD System</h1>
  <span class="chip">Planner → Test Architect → Implementer ⇄ Reviewer &nbsp;·&nbsp; model: {{model}} &nbsp;·&nbsp; max retries: {{max_retries}}</span>
</header>
<main>
  <textarea id="problem" placeholder="Describe the function to build… e.g. Implement two_sum(nums, target) returning the indices of the two numbers that add up to target."></textarea>
  <div class="row">
    <button id="run" onclick="start()">🚀 Run 4-Agent TDD</button>
    <span class="verdict" id="verdict" style="display:none"></span>
  </div>
  <div class="row">
    <span class="ex" onclick="ex(this)">Implement two_sum(nums: list[int], target: int) -> list[int] returning indices of the two numbers adding to target.</span>
    <span class="ex" onclick="ex(this)">Implement max_subarray(nums: list[int]) -> int returning the largest contiguous subarray sum.</span>
    <span class="ex" onclick="ex(this)">Implement is_balanced(s: str) -> bool checking balanced (), [], {} brackets.</span>
  </div>

  <div class="label">Pipeline status</div>
  <div id="status"><span class="empty">Idle. Press “Run 4-Agent TDD”.</span></div>

  <div class="label">What each agent is writing — live</div>
  <div class="agents">
    <div class="panel" id="p-planner">
      <div class="phead">🧠 Planner <span class="sub">writes spec.md</span><span class="dot" id="dot-planner"></span></div>
      <div class="feed" id="feed-planner"><div class="empty">waiting…</div></div>
    </div>
    <div class="panel" id="p-architect">
      <div class="phead">🧪 Test Architect <span class="sub">writes fuzz tests</span><span class="dot" id="dot-architect"></span></div>
      <div class="feed" id="feed-architect"><div class="empty">waiting…</div></div>
    </div>
    <div class="panel" id="p-implementer">
      <div class="phead">💻 Implementer <span class="sub">writes solution.py</span><span class="dot" id="dot-implementer"></span></div>
      <div class="feed" id="feed-implementer"><div class="empty">waiting…</div></div>
    </div>
    <div class="panel" id="p-reviewer">
      <div class="phead">👁️ Reviewer <span class="sub">turns errors → fixes</span><span class="dot" id="dot-reviewer"></span></div>
      <div class="feed" id="feed-reviewer"><div class="empty">waiting…</div></div>
    </div>
  </div>

  <div class="label">Final artifacts</div>
  <div class="card">
    <div class="tabs">
      <span class="tab active" data-k="solution" onclick="tab('solution')">solution.py</span>
      <span class="tab" data-k="tests" onclick="tab('tests')">test_solution.py</span>
      <span class="tab" data-k="spec" onclick="tab('spec')">spec.md</span>
      <span class="tab" data-k="output" onclick="tab('output')">test output</span>
    </div>
    <pre id="solution"><span class="empty">The final solution appears here.</span></pre>
    <pre id="tests" style="display:none"><span class="empty">The fuzz tests appear here.</span></pre>
    <pre id="spec" style="display:none"><span class="empty">The spec appears here.</span></pre>
    <pre id="output" style="display:none"><span class="empty">The pytest output appears here.</span></pre>
  </div>
</main>
<script>
let es=null;
const AGENTS=['planner','architect','implementer','reviewer'];
function ex(e){document.getElementById('problem').value=e.textContent;}
function tab(k){document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active',t.dataset.k===k));
  ['solution','tests','spec','output'].forEach(id=>document.getElementById(id).style.display=id===k?'block':'none');}
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function badge(m){if(m.includes('✅'))return' badge-ok';if(m.includes('🛑')||m.includes('❌'))return' badge-bad';
  if(m.includes('⚠️')||m.includes('🔧'))return' badge-warn';return'';}
function setDot(role,on){AGENTS.forEach(a=>{const d=document.getElementById('dot-'+a);if(d)d.classList.toggle('active',a===role&&on);});}

function start(){
  const problem=document.getElementById('problem').value.trim();
  if(!problem){alert('Enter a problem first.');return;}
  document.getElementById('run').disabled=true;
  document.getElementById('verdict').style.display='none';
  document.getElementById('status').innerHTML='';
  AGENTS.forEach(a=>document.getElementById('feed-'+a).innerHTML='<div class="empty">waiting…</div>');
  ['solution','tests','spec','output'].forEach(id=>document.getElementById(id).textContent='');
  fetch('/run_tdd',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({problem})})
    .then(r=>r.json()).then(res=>{
      if(res.error){addStatus('❌ '+res.error);document.getElementById('run').disabled=false;return;}
      open();
    });
}
function addStatus(m){const s=document.getElementById('status');const d=document.createElement('span');
  d.className='st'+badge(m);d.textContent=m;s.appendChild(d);s.scrollTop=s.scrollHeight;}
function addStep(ev){
  if(ev.msg_role==='system'||ev.msg_role==='user')return;          // hide role prompts + observations echo
  const feed=document.getElementById('feed-'+ev.role);if(!feed)return;
  if(feed.querySelector('.empty'))feed.innerHTML='';
  setDot(ev.role,true);
  const d=document.createElement('div');d.className='step';
  let h='<div class="who">'+ev.role+(ev.msg_role==='tool'?' · output':' · thinking')+'</div>';
  if(ev.content) h+='<div class="txt">'+esc(ev.content.slice(0,700))+'</div>';
  (ev.commands||[]).forEach(c=>{if(c)h+='<div class="cmd">$ '+esc(c)+'</div>';});
  if(ev.msg_role==='tool'&&ev.returncode!=null){const ok=ev.returncode===0;
    h+='<span class="rc '+(ok?'ok':'bad')+'">exit '+ev.returncode+'</span>';}
  d.innerHTML=h;feed.appendChild(d);feed.scrollTop=feed.scrollHeight;
}
function open(){
  es=new EventSource('/tdd_stream');
  es.onmessage=(m)=>{const ev=JSON.parse(m.data);
    switch(ev.type){
      case 'status': addStatus(ev.message); break;
      case 'step': addStep(ev); break;
      case 'error': addStatus('❌ '+ev.message); break;
      case 'final':
        AGENTS.forEach(a=>setDot(a,false));
        document.getElementById('solution').textContent=ev.solution||'(none)';
        document.getElementById('tests').textContent=ev.tests||'(none)';
        document.getElementById('spec').textContent=ev.spec||'(none)';
        document.getElementById('output').textContent=ev.test_output||'(none)';
        const v=document.getElementById('verdict');v.style.display='inline-block';
        v.className='verdict '+(ev.passed?'ok':'bad');
        v.textContent=(ev.passed?'✅ PASS':'❌ FAIL')+' · '+ev.attempts+' retr'+(ev.attempts===1?'y':'ies');
        break;
      case 'end': es.close(); AGENTS.forEach(a=>setDot(a,false)); document.getElementById('run').disabled=false; break;
    }
  };
  es.onerror=()=>{es.close();AGENTS.forEach(a=>setDot(a,false));document.getElementById('run').disabled=false;};
}
</script></body></html>
"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)

import time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_agent_intent as R
from openai import OpenAI
from amrl.config import load_config
cfg = load_config(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml"), [])
cli = OpenAI(base_url=cfg.controller.endpoint, api_key="EMPTY"); model = cfg.controller.model
SYS = ("You are an operations agent for an online modulation-recognition system. " + R.TOOL_API +
       "\nGiven an operator request, output the MINIMAL set of tools needed. End with: TOOLS: <comma-separated>.")
intents = [b[0] for b in R.BANK if b[2] == "novel"][:10]
ts = []
for it in intents:
    t0 = time.time()
    cli.chat.completions.create(model=model, messages=[{"role": "system", "content": SYS}, {"role": "user", "content": it}],
                                temperature=0.0, max_tokens=120, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    ts.append(time.time() - t0)
ts.sort()
print("Qwen3-8B single-shot plan latency (n=%d): median=%.2fs p99=%.2fs mean=%.2fs" % (
    len(ts), ts[len(ts)//2], ts[-1], sum(ts)/len(ts)))

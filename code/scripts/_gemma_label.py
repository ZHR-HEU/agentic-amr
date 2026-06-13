import json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
TOOLS = ["check_acc", "detect_drift", "confusion_probe", "openset_reject", "adapt", "allocate_budget"]
d = json.load(open("/tmp/nov_intents.json"))
intents = d["intents"]  # list of [text, gt_tools]
prompt = (d["tool_api"] +
          "\nFor EACH operator request below, list the MINIMAL set of tools needed (only tool names from the list). "
          "Output exactly one line per request as 'N: tool1, tool2'. No explanation.\n\n" +
          "\n".join(f"{i}. {it[0]}" for i, it in enumerate(intents)))
mp = "models/gemma-4-12B-it"
tok = AutoTokenizer.from_pretrained(mp)
mdl = AutoModelForCausalLM.from_pretrained(mp, torch_dtype=torch.bfloat16, device_map="auto").eval()
msgs = [{"role": "user", "content": prompt}]
enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(mdl.device)
with torch.no_grad():
    out = mdl.generate(**enc, max_new_tokens=600, do_sample=False, pad_token_id=tok.eos_token_id)
txt = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
labels = {}
for ln in txt.splitlines():
    m = re.match(r"\s*(\d+)\s*[:.]\s*(.*)", ln)
    if m:
        labels[int(m.group(1))] = sorted({t for t in TOOLS if t in m.group(2)})
json.dump(labels, open("/tmp/gemma_labels.json", "w"))
print("Gemma labeled %d / %d intents" % (len(labels), len(intents)))

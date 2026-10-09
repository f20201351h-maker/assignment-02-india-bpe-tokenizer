"""English-weight sweep starting from the RELEASED weights (D-entrade used the pre-fine-tuning weights)."""
import json, sys
sys.path.insert(0, "src")
from search import run_one, fmt
final = json.load(open("tokenizer/final_config.json", encoding="utf-8"))
L = final["langs"]
w0 = {tuple(k.rsplit(".", 1)): v for k, v in final["weights"].items()}
for f in [1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0]:
    w = {k: (v * f if k[0] == "en" else v) for k, v in w0.items()}
    rec, _ = run_one("D-entrade2", final["config"], L, w, note=f"from released weights; en weight x{f}")
    print(fmt(rec), flush=True)

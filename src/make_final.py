"""Retrain the released tokenizer from tokenizer/final_config.json and save tokenizer.json.

    python src/make_final.py            # train + save + record composition / held-out numbers
    python src/make_final.py --check    # train and only compare with the saved tokenizer.json
"""
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from pipeline import DEFAULTS, evaluate, train  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = ROOT / "tokenizer" / "final_config.json"
OUT = ROOT / "tokenizer" / "tokenizer.json"

if __name__ == "__main__":
    final = json.loads(CFG.read_text(encoding="utf-8"))
    langs = final["langs"]
    weights = {tuple(k.rsplit(".", 1)): v for k, v in final["weights"].items()}
    tok, info = train(final["config"], langs, weights)
    text = tok.to_str(pretty=True)
    compact_sha = hashlib.sha256(tok.to_str().encode("utf-8")).hexdigest()
    if final.get("source_tokenizer_sha256"):
        same = compact_sha == final["source_tokenizer_sha256"]
        print(f"retrained == ledger run {final.get('source_run')}: {same}")
        if not same:
            sys.exit("retrained tokenizer differs from the ledger run it should reproduce")
    if "--check" in sys.argv:
        old = OUT.read_text(encoding="utf-8")
        same = old == text
        print("retrained tokenizer identical to tokenizer/tokenizer.json:", same)
        sys.exit(0 if same else 1)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    res = evaluate(tok, langs, heldout=True, akshara_check=True)
    final["config"] = {**DEFAULTS, **final["config"]}
    final["composition"] = {"special": info["n_specials"], "characters": info["n_base_chars"],
                            "akshara": info["n_stage1"], "learned": info["n_learned"]}
    final["heldout"] = {l: {"fertility": res["heldout"][l]["fertility"], "words": res["heldout"][l]["words"],
                            "tokens": res["heldout"][l]["tokens"],
                            "akshara_violations": res["heldout"][l]["akshara_violations"],
                            "unk": res["heldout"][l]["unk"]} for l in langs}
    final["training_result"] = {l: res["eval"][l]["fertility"] for l in langs}
    final["training_spread"] = res["spread"]
    final["tokenizer_sha256"] = hashlib.sha256(OUT.read_bytes()).hexdigest()
    final["compact_sha256"] = compact_sha
    CFG.write_text(json.dumps(final, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps({k: final[k] for k in ("composition", "training_result", "training_spread", "tokenizer_sha256")}, indent=1))

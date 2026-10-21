"""D0 — Build a mixed (mdCATH + ATLAS) paired-window manifest (NEW, 2026-10-19).

Concatenates per-dataset manifests that may have DIFFERENT frame intervals
(mdCATH = 1000 ps, ATLAS-100ps = 100 ps) into one manifest.  The paired dataset
derives ``time_ps`` from each record's own ``frame_interval_ps``, so mixing is
safe; this script additionally:
  * optionally over-samples the minority dataset (ATLAS) so it is not drowned,
  * optionally rescales the ATLAS ``paired`` horizon semantics by writing a
    sidecar ``frame_scale`` note (informational only; horizons are in frames and
    the *time* is always recomputed from frame_interval_ps by the dataset),
  * verifies train/val PROTEIN disjointness and prints a composition summary.

Nothing here modifies any existing manifest or the precise-coordinate pipeline;
it only writes a NEW jsonl.

Usage:
    python -m scripts.build_mixed_manifest \
        --mdcath-train artifacts/mdcath_manifest_200dom/train.jsonl \
        --mdcath-val   artifacts/mdcath_manifest_200dom/val.jsonl \
        --atlas-train  artifacts/atlas_manifest_100ps/train.jsonl \
        --atlas-val    artifacts/atlas_manifest_100ps/val.jsonl \
        --atlas-repeat 3 \
        --out-dir artifacts/mixed_manifest_100ps
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _read(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _keys(records: list[dict]) -> set[str]:
    return {str(r.get("protein_id") or r.get("domain_id") or "") for r in records}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mdcath-train", default="artifacts/mdcath_manifest_200dom/train.jsonl")
    p.add_argument("--mdcath-val", default="artifacts/mdcath_manifest_200dom/val.jsonl")
    p.add_argument("--atlas-train", default="artifacts/atlas_manifest_100ps/train.jsonl")
    p.add_argument("--atlas-val", default="artifacts/atlas_manifest_100ps/val.jsonl")
    p.add_argument("--atlas-repeat", type=int, default=1,
                   help="repeat ATLAS rows N times (oversample minority)")
    p.add_argument("--out-dir", default="artifacts/mixed_manifest_100ps")
    p.add_argument("--tag", default="mixed_100ps",
                   help="informational tag stored in each record's metadata")
    args = p.parse_args()

    if args.atlas_repeat < 1:
        raise ValueError("--atlas-repeat must be >= 1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def build(split, mdc, atl, repeat):
        md = _read(mdc)
        at = _read(atl)
        at_sup = []
        for i in range(repeat):
            for r in at:
                rr = dict(r)
                rr.setdefault("metadata", {})
                rr["metadata"] = dict(rr["metadata"])
                rr["metadata"]["mix_tag"] = args.tag
                rr["metadata"]["atlas_copy"] = i
                at_sup.append(rr)
        recs = md + at_sup
        # disjointness: mdCATH protein ids vs ATLAS protein ids must not intersect
        mk, ak = _keys(md), _keys(at)
        overlap = mk & ak
        head = out_dir / f"{split}.jsonl"
        with head.open("w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        comp = Counter(r.get("dataset") for r in recs)
        temps = Counter((r.get("dataset"), r.get("temperature_k")) for r in recs)
        print(f"[{split}] wrote {len(recs)} records -> {head}")
        print(f"    datasets: {dict(comp)}")
        print(f"    train/ATLAS vs mdCATH protein overlap: {sorted(overlap) or 'EMPTY'}")
        for k, v in sorted(temps.items(), key=lambda x: str(x[0])):
            print(f"    {k}: {v}")
        return recs, mk, ak

    tr, mk_tr, ak_tr = build("train", args.mdcath_train, args.atlas_train, args.atlas_repeat)
    # val is NEVER oversampled (would double-count in bootstrap CI)
    va, mk_va, ak_va = build("val", args.mdcath_val, args.atlas_val, 1)

    # cross-split disjointness (train vs val proteins, per dataset)
    print("\n=== cross-split protein disjointness ===")
    for name, a, b in (("mdCATH", mk_tr, mk_va), ("ATLAS", ak_tr, ak_va)):
        ov = a & b
        print(f"  {name}: train∩val proteins = {sorted(ov) or 'EMPTY'} "
              f"({'OK disjoint' if not ov else 'LEAK!'})")

    meta = {
        "tag": args.tag,
        "atlas_repeat": args.atlas_repeat,
        "mdcath_train": args.mdcath_train, "mdcath_val": args.mdcath_val,
        "atlas_train": args.atlas_train, "atlas_val": args.atlas_val,
        "n_train": len(tr), "n_val": len(va),
    }
    (out_dir / "mix_info.json").write_text(json.dumps(meta, indent=1))
    print(f"\nwrote {out_dir/'mix_info.json'}")


if __name__ == "__main__":
    main()

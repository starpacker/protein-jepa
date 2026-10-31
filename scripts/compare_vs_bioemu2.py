"""Summarise any Lasp-variant result json against Model 1 and bioemu2.

Reads a `scripts/eval_lasp_final.py` output and prints, per horizon, whether the
variant beats bioemu2 and Model 1 on each metric, plus the win/loss totals.
"""
import json, sys, numpy as np
M=["rmsf_corr","rmsf_mae","root_mean_w2","variance_w2","md_pca_w2","weak_contact_j","transient_contacts_j"]
HB={"rmsf_corr","weak_contact_j","transient_contacts_j"}
def bk(hz):
    f=float(hz); return str(int(f)) if f==int(f) else str(f)

def load(path):
    return json.load(open(path))

def main():
    lasp=load(sys.argv[1])
    bio=load("results/bioemu2_reps/summary.json")["by_horizon"]
    name=sys.argv[2] if len(sys.argv)>2 else "Lasp"
    wb=lb=wm=lm=0
    print(f"### {name} vs bioemu2 / Model 1 ({sys.argv[1]})\n")
    hdr=f"{'hz':>5} {'metric':22}{'bioemu2':>9}{'Model1':>9}{name:>9}   {'vsBio':>6}{'vsM1':>6} sig"
    print(hdr); print("-"*len(hdr))
    for hz in sorted(lasp["by_horizon"], key=float):
        rows=lasp["by_horizon"][hz]; b=bio[bk(hz)]["metrics"]
        for m in M:
            r=rows[m]; bv=b[m]["mean"]; hb=m in HB
            beat_bio=(r["lasp"]>bv) if hb else (r["lasp"]<bv)
            beat_m1=(r["lasp"]>r["model1"]) if hb else (r["lasp"]<r["model1"])
            wb+=beat_bio; lb+=not beat_bio; wm+=beat_m1; lm+=not beat_m1
            print(f"{hz:>5} {m:22}{bv:>9.4f}{r['model1']:>9.4f}{r['lasp']:>9.4f}"
                  f"{' *' if r['significant'] else '  '} {('WIN' if beat_bio else 'lose'):>6}"
                  f"{('WIN' if beat_m1 else 'lose'):>6}")
        print()
    print(f"TOTAL vs bioemu2: {wb}W/{lb}L    vs Model 1: {wm}W/{lm}L   (out of {wb+lb})")
    print(f"BEATS BOTH = {'YES' if (wb and lb==0) else 'NO (not on every cell)'}")
    print(f"  vs bioemu2 >=50% = {wb > (wb+lb)/2}")

if __name__=="__main__":
    main()

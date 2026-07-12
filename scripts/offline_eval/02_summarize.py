#!/usr/bin/env python
"""Collect vf-eval result JSONLs -> policy x family table (+ ops columns)."""
import json, glob, os, statistics as st
here = os.path.dirname(os.path.abspath(__file__))
outs = sorted(glob.glob(os.path.join(here, '..', '..', 'outputs', 'offline_eval_*')))
assert outs, 'no offline_eval_* outputs found'
OUT = outs[-1]
rows = {}
for f in glob.glob(f'{OUT}/*/*/**/*.jsonl', recursive=True):
    rel = os.path.relpath(f, OUT).split(os.sep)
    fam, pol = rel[0], rel[1]
    for line in open(f):
        try: r = json.loads(line)
        except Exception: continue
        rew = r.get('reward'); m = r.get('metrics') or {}
        if rew is None: continue
        d = rows.setdefault((fam, pol), {'rew': [], 'turns': [], 'sub': []})
        d['rew'].append(float(rew))
        if 'rlm_iterations' in m: d['turns'].append(float(m['rlm_iterations']))
        if 'rlm_sub_llm_calls' in m: d['sub'].append(float(m['rlm_sub_llm_calls']))
print(f'{"family":26s} {"policy":42s} {"n":>4s} {"acc":>6s} {"SE":>5s} {"turns":>6s} {"sub":>7s}')
for (fam, pol), d in sorted(rows.items()):
    n = len(d['rew']); acc = st.mean(d['rew'])
    se = (acc*(1-acc)/n)**0.5 if 0 < acc < 1 else 0.0
    t = st.mean(d['turns']) if d['turns'] else float('nan')
    s = st.mean(d['sub']) if d['sub'] else float('nan')
    print(f'{fam:26s} {pol:42s} {n:4d} {acc:6.3f} {se:5.3f} {t:6.2f} {s:7.1f}')
print('\nnext: per-rollout JSONLs -> ab_paired_cost.py (both-correct paired estimand)')

import argparse

import yaml

from hmm_agents.config import load_config
from hmm_agents.runner import MODES, rescore, run_experiment

ap = argparse.ArgumentParser()
ap.add_argument("experiment")
ap.add_argument("--mode", required=True, choices=MODES)
ap.add_argument("--final", action="store_true")
ap.add_argument("--config", default=None)
ap.add_argument("--n-boot", type=int, default=None, help="override eval.n_boot (quick checks)")
ap.add_argument("--rescore", action="store_true",
                help="re-score saved predictions (after a harness change) without refitting")
args = ap.parse_args()

with open(args.experiment, encoding="utf-8") as f:
    exp = yaml.safe_load(f)
cfg = load_config(args.config)
if args.rescore:
    if args.mode == "test":
        raise SystemExit("--rescore is not allowed on test")
    out = rescore(exp, cfg, args.mode, n_boot=args.n_boot)
else:
    out = run_experiment(exp, cfg, args.mode, final=args.final, n_boot=args.n_boot)
print(f"\nWrote {out}/")

import argparse

from hmm_agents.config import load_config
from hmm_agents.data import download_shards

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--all", action="store_true", help="download all 12 shards (~1.1 GB)")
args = ap.parse_args()

cfg = load_config(args.config)
shards = None if args.all else cfg["data"]["shards"]
paths = download_shards(cfg["data"]["repo_id"], cfg["paths"]["raw"], shards)
print(f"Downloaded {len(paths)} shard(s) to {cfg['paths']['raw']}")

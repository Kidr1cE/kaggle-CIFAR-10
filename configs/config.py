import yaml
import argparse

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./conf/default.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config

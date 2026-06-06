import argparse
from core.config import load_config
from core.runner import run_experiment

def main():
    parser = argparse.ArgumentParser(description="VLM Benchmark Runner")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_experiment(cfg)

if __name__ == "__main__":
    main()
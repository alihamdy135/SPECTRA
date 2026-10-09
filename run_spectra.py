import argparse

from ablate import run_ablations
from config import load_config
from evaluate import run_main_comparison
from runtime import build_evaluation_runtime, build_training_runtime
from train import train_adapters
from utils import configure_run, print_sep, set_seed


def run_spectra(config_path="configs/experiment.yaml"):
    config = load_config(config_path)
    logger = configure_run(config, "run")
    set_seed(config.seed)
    model, train_sets, adapt_sets, test_sets = build_training_runtime(config, logger)
    print_sep("Metrics")
    print("  EM, F1, ROUGE-L, Composite(0.40/0.35/0.25) + Bootstrap CI")
    train_adapters(model, train_sets, config, logger)
    runtime = build_evaluation_runtime(
        config,
        logger,
        model=model,
        split_data=(train_sets, adapt_sets, test_sets),
    )
    run_main_comparison(runtime)
    return run_ablations(runtime=runtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    arguments = parser.parse_args()
    run_spectra(arguments.config)


if __name__ == "__main__":
    main()

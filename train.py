import argparse

from config import load_config
from models.lora import train_domain_adapter
from runtime import build_training_runtime
from utils import Timer, configure_run, set_seed, print_sep


def train_adapters(model, train_sets, config, logger):
    adapters = {}
    print_sep("LoRA Training")
    with Timer("LoRA"):
        for domain in config.lora_names:
            adapters[domain] = train_domain_adapter(
                model,
                train_sets[config.domain_map.get(domain, domain)],
                domain,
                config,
                logger,
            )
    return adapters


def run_training(config_path):
    config = load_config(config_path)
    logger = configure_run(config, "train")
    set_seed(config.seed)
    model, train_sets, _, _ = build_training_runtime(config, logger)
    return train_adapters(model, train_sets, config, logger)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    arguments = parser.parse_args()
    run_training(arguments.config)


if __name__ == "__main__":
    main()

import os
from pathlib import Path

import hydra
import omegaconf

import env

# Load environment variables
env.load_envs()

# Set the cwd to the project root
PROJECT_ROOT: Path = Path(env.get_env("PROJECT_ROOT"))
assert (
    PROJECT_ROOT.exists()
), "You must configure the PROJECT_ROOT environment variable in a .env file!"

os.chdir(PROJECT_ROOT)


def run_hoa_predictor(cfg: omegaconf.DictConfig):
    from source.hoa_predictor import HOAPredictor

    ...


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    run_hoa_predictor(cfg)
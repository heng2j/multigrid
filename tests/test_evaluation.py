import pytest

import argparse
import os
from pathlib import Path
from multigrid.scripts.visualize import main_evaluation
from multigrid.utils.training_utilis import get_checkpoint_dir
from multigrid.envs import CONFIGURATIONS
import json


SUBMISSION_CONFIG_FILE = sorted(
    Path("submission").expanduser().glob("**/submission_config.json"), key=os.path.getmtime
)[-1]

with open(SUBMISSION_CONFIG_FILE, "r") as file:
    submission_config_data = file.read()

submission_config = json.loads(submission_config_data)

SUBMITTER_NAME = submission_config["name"]
SAVE_DIR = "submission/evaluation_reports/from_github_actions"


def test_evaluation():
    # Create/check paths
    search_dir = "submission/ray_results"
    assert os.path.exists(search_dir), f"Directory {search_dir} does not exist!"

    checkpoint_dirs = [
        checkpoint_dir.parent
        for checkpoint_dir in sorted(Path(search_dir).expanduser().glob("**/result.json"), key=os.path.getmtime)
    ]

    checkpoint_paths = [
        sorted(Path(checkpoint_dir).expanduser().glob("**/*.is_checkpoint"), key=os.path.getmtime)[-1].parent
        for checkpoint_dir in checkpoint_dirs
    ]

    for checkpoint_path in checkpoint_paths:
        # Define parameters for the test
        env = str(checkpoint_path).split("/")[-2].split("_")[1] + "-Eval"
        training_scheme = CONFIGURATIONS[env][1]["training_scheme"]
        teams = CONFIGURATIONS[env][1]["teams"]
        scenario_name = env.split("-v3-")[1]
        gif = f"{SAVE_DIR}/{scenario_name}_{SUBMITTER_NAME}"

        # Set argument
        params = {
            "algo": "PPO",
            "framework": "torch",
            "lstm": False,
            "env": env,
            "env_config": {},
            "num_agents": sum(teams.values()),
            "num_episodes": 10,
            "load_dir": checkpoint_path,
            "gif": gif,
            "our_agent_ids": [0, 1],
            "teams": teams,
            "training_scheme": training_scheme,
            "render_mode": "rgb_array",
            "save_dir": SAVE_DIR,
        }

        args = argparse.Namespace(**params)

        # Call the evaluation function
        main_evaluation(args)

        # Check the generated evaluation reports
        eval_report_path = os.path.join(args.save_dir, "eval_summary.csv")
        assert os.path.exists(eval_report_path), f"Expected evaluation report {eval_report_path} doesn't exist!"

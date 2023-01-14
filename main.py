from pathlib import Path

import sys

from experiments import Experiment
from gpu import wait_and_set_gpu_to_next_available


def run(config_file_path: Path):
	wait_and_set_gpu_to_next_available()
	exp = Experiment(conf_path=config_file_path)
	exp.run()


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print(f'usage: {sys.argv[0]} <exp.yaml>', file=sys.stderr)
		print(f'\texp.yam: experiment configuration in yaml format', file=sys.stderr)
		exit(1)

	exp_config_path = Path(sys.argv[1])
	if not exp_config_path.exists():
		raise FileNotFoundError(f'Experiment configuration file not found: {exp_config_path}')

	run(exp_config_path)

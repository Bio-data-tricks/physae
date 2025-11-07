.PHONY: setup lint test train infer

setup:
python -m pip install -e .[dev]

lint:
pre-commit run --all-files

test:
pytest

train:
python -m physae.cli.train --config configs/train.yaml

infer:
python -m physae.cli.infer --config configs/infer.yaml --checkpoint outputs/latest.ckpt --input data/sample_input.json --output outputs/prediction.json

.PHONY: setup-submodule setup-env setup-upstream short-clip parity test build-wheel cli

setup-submodule:
	./scripts/setup_submodule.sh

setup-env:
	./scripts/setup_env.sh

setup-upstream:
	./scripts/setup_upstream.sh

short-clip:
	@test -n "$(INPUTS)" || (echo "Usage: make short-clip INPUTS=\"/path/to/input_a /path/to/input_b\"" && exit 1)
	./scripts/make_short_clip.sh $(INPUTS)

parity:
	UPSTREAM_PARITY_ASSETS=1 pytest tests/test_parity_upstream.py

test:
	pytest

build-wheel:
	python3 -m build --no-isolation

cli:
	python3 -m madmom_beats_lite.cli --help

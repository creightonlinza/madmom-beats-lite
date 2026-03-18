.PHONY: setup-submodule setup-env short-clip golden-short golden-all parity test build-wheel cli

setup-submodule:
	./scripts/setup_submodule.sh

setup-env:
	./scripts/setup_env.sh

short-clip:
	@test -n "$(INPUTS)" || (echo "Usage: make short-clip INPUTS=\"/path/to/input_a /path/to/input_b\"" && exit 1)
	./scripts/make_short_clip.sh $(INPUTS)

golden-short:
	./scripts/generate_golden.sh short

golden-all:
	./scripts/generate_golden.sh all

parity:
	./scripts/run_parity.sh

test:
	pytest

build-wheel:
	python3 -m build --no-isolation

cli:
	python3 -m madmom_beats_lite.cli --help

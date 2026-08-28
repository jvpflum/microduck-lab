SHELL := /bin/bash

.PHONY: bootstrap preflight test list-envs smoke skate-smoke verify-artifact \
	verify-skate-artifact train-baseline train-skate verify

bootstrap:
	./scripts/bootstrap.sh

preflight:
	./scripts/preflight.sh

test:
	./scripts/test.sh

list-envs:
	./scripts/ducklab.sh list-envs

smoke:
	./scripts/smoke.sh

skate-smoke:
	./scripts/skate-smoke.sh

verify-artifact:
	./scripts/verify-artifact.sh

verify-skate-artifact:
	./scripts/verify-skate-artifact.sh

train-baseline:
	./scripts/train-baseline.sh

train-skate:
	./scripts/train-skate.sh

verify: preflight test smoke verify-artifact

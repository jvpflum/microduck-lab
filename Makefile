SHELL := /bin/bash

.PHONY: bootstrap preflight test list-envs smoke verify-artifact train-baseline verify

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

verify-artifact:
	./scripts/verify-artifact.sh

train-baseline:
	./scripts/train-baseline.sh

verify: preflight test smoke verify-artifact

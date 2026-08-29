SHELL := /bin/bash

.PHONY: bootstrap preflight test test-gamepad list-envs smoke skate-smoke swizzle-smoke \
	verify-artifact verify-skate-artifact evaluate-swizzle train-baseline train-skate \
	train-swizzle verify

bootstrap:
	./scripts/bootstrap.sh

preflight:
	./scripts/preflight.sh

test:
	./scripts/test.sh

test-gamepad:
	./scripts/test-gamepad.sh

list-envs:
	./scripts/ducklab.sh list-envs

smoke:
	./scripts/smoke.sh

skate-smoke:
	./scripts/skate-smoke.sh

swizzle-smoke:
	./scripts/swizzle-smoke.sh

verify-artifact:
	./scripts/verify-artifact.sh

verify-skate-artifact:
	./scripts/verify-skate-artifact.sh

evaluate-swizzle:
	./scripts/evaluate-swizzle.sh

train-baseline:
	./scripts/train-baseline.sh

train-skate:
	./scripts/train-skate.sh

train-swizzle:
	./scripts/train-swizzle.sh

verify: preflight test smoke verify-artifact

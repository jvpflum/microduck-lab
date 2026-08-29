SHELL := /bin/bash

.PHONY: bootstrap build-pollen-arena preflight test test-gamepad test-policy-bench list-envs smoke skate-smoke swizzle-smoke hop-smoke \
	verify-artifact verify-skate-artifact evaluate-swizzle train-baseline train-skate \
	train-swizzle train-hop import-pollen-baselines bench-discover bench-list bench-dashboard bench-metrics bench-score bench-star verify

bootstrap:
	./scripts/bootstrap.sh

build-pollen-arena:
	./scripts/build-pollen-arena.sh

preflight:
	./scripts/preflight.sh

test:
	./scripts/test.sh

test-gamepad:
	./scripts/test-gamepad.sh

test-policy-bench:
	./scripts/test-policy-bench.sh

list-envs:
	./scripts/ducklab.sh list-envs

smoke:
	./scripts/smoke.sh

skate-smoke:
	./scripts/skate-smoke.sh

swizzle-smoke:
	./scripts/swizzle-smoke.sh

hop-smoke:
	./scripts/hop-smoke.sh

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

train-hop:
	./scripts/train-hop.sh

import-pollen-baselines:
	./scripts/import-pollen-baselines.sh

bench-discover:
	./scripts/policy-bench.sh discover

bench-list:
	./scripts/policy-bench.sh list

bench-dashboard:
	./scripts/serve-policy-bench.sh

bench-metrics:
	@test -n "$(RUN)" || (echo 'Usage: make bench-metrics RUN=<run-id>' >&2; exit 2)
	./scripts/policy-bench.sh metrics "$(RUN)"

bench-score:
	@test -n "$(RUN)" || (echo 'Usage: make bench-score RUN=<run-id>' >&2; exit 2)
	./scripts/policy-bench.sh score "$(RUN)"

bench-star:
	@test -n "$(RUN)" || (echo 'Usage: make bench-star RUN=<run-id>' >&2; exit 2)
	./scripts/policy-bench.sh star "$(RUN)" --note "$(NOTE)"

verify: preflight test smoke verify-artifact

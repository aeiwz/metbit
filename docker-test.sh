#!/usr/bin/env bash
# docker-test.sh — build and run the metbit test container for one or all Python versions.
#
# Usage:
#   ./docker-test.sh                   # test all versions (3.10 → 3.13)
#   ./docker-test.sh 3.11              # single version
#   ./docker-test.sh 3.12 -k pca      # filter tests
#   ./docker-test.sh all               # explicitly test all versions
#   KEEP=1 ./docker-test.sh 3.12       # keep container after run

set -euo pipefail

ALL_VERSIONS=("3.10" "3.11" "3.12" "3.13")

run_version() {
    local PY="$1"
    shift
    local IMAGE="metbit-test-py${PY//.}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Python ${PY}  →  ${IMAGE}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    echo "--> Building..."
    docker build \
        --build-arg PYTHON_VERSION="${PY}" \
        -f Dockerfile.test \
        -t "${IMAGE}" \
        . \
        --quiet

    local DOCKER_OPTS="--rm"
    if [[ "${KEEP:-0}" == "1" ]]; then
        DOCKER_OPTS=""
        echo "    (container kept — find it with 'docker ps -a')"
    fi

    echo "--> Running tests..."
    if docker run ${DOCKER_OPTS} "${IMAGE}" \
            pytest -m "not slow and not perf" --tb=short -q "$@"; then
        echo "✓  Python ${PY} PASSED"
        return 0
    else
        echo "✗  Python ${PY} FAILED"
        return 1
    fi
}

# Determine what to run
ARG="${1:-all}"
shift || true

if [[ "$ARG" == "all" ]]; then
    FAILED=()
    for PY in "${ALL_VERSIONS[@]}"; do
        run_version "$PY" "$@" || FAILED+=("$PY")
    done

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if [[ ${#FAILED[@]} -eq 0 ]]; then
        echo "  All versions PASSED: ${ALL_VERSIONS[*]}"
    else
        echo "  FAILED versions: ${FAILED[*]}"
        exit 1
    fi
else
    run_version "$ARG" "$@"
fi

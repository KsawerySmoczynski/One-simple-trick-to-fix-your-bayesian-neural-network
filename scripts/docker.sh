#!/bin/bash
MODE=$1
DEVICE=$2

source scripts/utils.sh || true

validate_arg "${MODE}" MODE
validate_arg "${DEVICE}" DEVICE
{
case ${MODE} in
    build)
        validate_arg "${DEVICE}" DEVICE
        export DOCKER_BUILDKIT=1
        docker build --progress=plain --build-arg=DEVICE=${DEVICE} -t bayesian-${DEVICE} . ;;
    run)
        docker run -dit bayesian-${DEVICE} /bin/bash ;;
    *)
        echo "Unsupported mode"
esac
} 2>&1 | tee dockerbuild.log

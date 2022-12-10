#!/bin/bash
set -a

function validate_arg(){
    ARG=$1
    ARGNAME=$2
    if [ -z "${ARG}" ]; then
        echo "${ARGNAME} not provided"
        exit 0
    fi
}

set +a

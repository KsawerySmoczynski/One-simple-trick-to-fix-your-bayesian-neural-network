ARG DEVICE
FROM python:3.8 as cpu
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04 as gpu
RUN apt-get update \
    && apt-get install -y --no-install-recommends  \
        git curl python3.8 python3.8-dev python3.8-distutils \
        python-is-python3 \
    && ln -sf /usr/bin/python3.8 /usr/bin/python \
    && ln -sf /usr/bin/python3.8 /usr/bin/python3
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3 get-pip.py \
    && ln -sf /usr/bin/pip3.8 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* get-pip.py

FROM ${DEVICE} as image

RUN pip install --no-cache --upgrade pip wheel setuptools poetry

COPY poetry.lock pyproject.toml /src/ /app/

WORKDIR app

RUN  \
    --mount=type=cache,target=/root/.cache/pypoetry/cache \
    --mount=type=cache,target=/root/.cache/pypoetry/artifacts \
    poetry install --no-root

RUN poetry run pip install setuptools==59.5.0

COPY . .

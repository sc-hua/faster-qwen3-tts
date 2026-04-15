ARG BASE_IMAGE=python:3.12-slim-bookworm
# can be overridden by build arg, e.g. --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

FROM ${BASE_IMAGE}

# can be overridden by build arg, e.g. --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_INDEX_URL

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 

WORKDIR /app

# can be overridden by build arg, e.g. --build-arg APT_MIRROR=mirrors.tuna.tsinghua.edu.cn
ARG APT_MIRROR=deb.debian.org

RUN set -eux; \
    for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do \
        [ -f "$f" ] || continue; \
        sed -i \
            -e "s|deb.debian.org|${APT_MIRROR}|g" \
            -e "s|security.debian.org|${APT_MIRROR}|g" \
            -e "s|archive.ubuntu.com|${APT_MIRROR}|g" \
            -e "s|security.ubuntu.com|${APT_MIRROR}|g" \
            "$f"; \
    done; \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY . /app


FROM debian:bookworm-slim as builder

WORKDIR /opt

ENV RYE_HOME="/opt/rye"
ENV PATH="$RYE_HOME/shims:$PATH"

# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    curl

SHELL [ "/bin/bash", "-o", "pipefail", "-c" ]
RUN curl -sSf https://rye-up.com/get | RYE_INSTALL_OPTION="--yes" bash && \
    rye config --set-bool behavior.global-python=true && \
    rye config --set-bool behavior.use-uv=true

COPY ./.python-version ./pyproject.toml ./requirements* ./
RUN rye pin "$(cat .python-version)" && \
    rye sync --no-dev


FROM debian:bookworm-slim
COPY --from=builder /opt/rye /opt/rye

ENV RYE_HOME="/opt/rye"
ENV PATH="$RYE_HOME/shims:$PATH"
ENV PYTHONUNBUFFERED True

RUN apt-get -y update \
    && apt-get install --no-install-recommends -y ffmpeg=7:5.1.4-0+deb12u1\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

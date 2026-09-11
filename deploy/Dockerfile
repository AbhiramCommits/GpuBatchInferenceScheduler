# syntax=docker/dockerfile:1.7
# Multi-stage build: CUDA 12.4 devel image compiles the C++ targets and the
# Python wheel; the slim runtime image ships only the binaries + wheel.

ARG CUDA_VERSION=12.4.0

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      cmake \
      ninja-build \
      git \
      g++ \
      python3 \
      python3-dev \
      python3-pip \
      python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# Python wheel (scikit-build-core drives CMake + CUDA through pyproject.toml)
RUN python3 -m pip install --no-cache-dir scikit-build-core \
    && python3 -m pip wheel --no-deps --no-cache-dir -w /wheels .

# C++ binaries: benchmark harness + scheduler daemon
RUN cmake -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="70;80;89" \
      -DGBIS_BUILD_PYTHON_BINDINGS=OFF \
      -DGBIS_BUILD_TESTS=OFF \
    && cmake --build build -j \
    && cmake --install build --prefix /out

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      python3 \
      python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -r -u 1000 -m gbis

COPY --from=builder /out/bin/bench_gemm /usr/local/bin/bench_gemm
COPY --from=builder /out/bin/scheduler_daemon /usr/local/bin/scheduler_daemon
COPY --from=builder /wheels/ /wheels/

RUN python3 -m pip install --no-cache-dir --no-deps /wheels/*.whl numpy \
    && rm -rf /wheels

USER gbis
WORKDIR /home/gbis
EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/scheduler_daemon"]
CMD ["--jobs", "1000000", "--arrival-rate", "20", "--http-port", "8080"]

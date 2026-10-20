# Deployment: the identical workload, three ways

## Validation status

Validated on a Tesla T4 Colab GPU VM on 2026-10-20. **Only the bare-host
binary path was validated.** The container, Kubernetes, and Slurm paths are
**untested** in this environment and are not claimed to work.

| Path | Status | What happened |
|---|---|---|
| Bare host binary | **Validated** | `bench_gemm`, the test suites, sanitizers, scheduler stress, and profiling all ran directly on the host. |
| Docker container | **Untested** | The Docker daemon starts, but this VM is itself a container with a threaded cgroup hierarchy. `docker build` fails at the first `RUN` with `unable to apply cgroup configuration ... cgroup.procs`, and `docker run` fails with `failed to create default sandbox: operation not permitted`. The image was not built or run; `/healthz`, `/metrics`, and in-container `bench_gemm` were not validated. |
| Kubernetes | **Untested** | No `kubectl`, k3s, or microk8s is present, and the container runtime cannot create containers (same cgroup restriction). The NVIDIA device plugin, GPU allocation, probes, and Job were not validated. |
| Slurm | **Untested** | `slurmctld` starts, but `slurmd` cannot initialize: Slurm 23.11's cgroup/v2 plugin requires systemd/D-Bus cgroup scopes, which this container does not have. `sbatch --test-only` fails with `Requested node configuration is not available`, and no Slurm job was executed. |

The commands below are the intended usage; they have **not** been validated
end-to-end in this environment.

Every deployment path in this directory runs the **same workload**: the
batched GEMM benchmark `C[b] = A[b] * B[b]` with `M=N=K=1024`, batch `8`,
float32, 10 iterations, via the cuBLAS path
(`bench_gemm --m 1024 --n 1024 --k 1024 --batch 8 --iters 10 --impl cublas`).
Wherever it runs, it prints one CSV row per measurement set
(impl,m,n,k,batch,iters,h2d_ms,kernel_ms,d2h_ms,total_ms,gflops,
cpu_gflops,speedup_vs_cpu,max_abs_err), which makes the three execution
environments directly comparable.

| Way | Where | Runner | Results |
|-----|-------|--------|---------|
| 1. Bare host binary | workstation / Slurm compute node | `./build/bench_gemm` | **Validated** (stdout) |
| 2. Container | workstation (docker) or Slurm via srun + Pyxis/enroot | `docker run --gpus all ...` / `srun --container-image=...` | **Untested** in this environment |
| 3. Kubernetes | GPU cluster (`nvidia.com/gpu: 1`) | `kubectl apply -k deploy/k8s` | **Untested** in this environment |
| Slurm | GPU cluster | `sbatch deploy/slurm/submit.sbatch` | **Untested** in this environment |

The long-running scheduler daemon (`scheduler_daemon`) ships in the same
image; it is deployed by `deploy/k8s/deployment.yaml` and can be probed on
`GET /healthz` and `GET /metrics`.

## Build the image

> **Status: untested.** The image was not built in this environment: the
> nested container's threaded cgroup hierarchy prevents `runc` from creating
> the build container. The CUDA arch list was updated to include `sm_75` for
> the Tesla T4 target, but this change is not runtime-validated.

```sh
docker build -t gpuinfer/scheduler:latest -f deploy/Dockerfile .
```

The multi-stage Dockerfile builds the C++ binaries and the `gpuinfer` Python
wheel in `nvidia/cuda:12.4.0-devel-ubuntu22.04` and copies only the binaries,
the wheel, and Python+numpy into the slim `runtime` image (non-root user,
ENTRYPOINT = scheduler daemon).

## 1. Bare host binary

```sh
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGBIS_BUILD_TESTS=OFF
cmake --build build -j
./build/bench_gemm --m 1024 --n 1024 --k 1024 --batch 8 --iters 10 --impl cublas
```

## 2. Container

> **Status: untested.** `docker run` could not create a container in this
> environment (`failed to create default sandbox: operation not permitted`),
> so the daemon probes, `/metrics`, and in-container `bench_gemm` were not
> exercised. The commands below are the intended usage, not validated here.

Local (requires the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)):

```sh
docker run --rm --gpus all --entrypoint bench_gemm \
  gpuinfer/scheduler:latest \
  --m 1024 --n 1024 --k 1024 --batch 8 --iters 10 --impl cublas
```

Run the daemon with its HTTP status server:

```sh
docker run --rm --gpus all -p 8080:8080 \
  gpuinfer/scheduler:latest \
  scheduler_daemon --jobs 1000 --arrival-rate 20 --http-port 8080
curl http://localhost:8080/healthz
curl http://localhost:8080/metrics
```

On Slurm with Pyxis/enroot (falls back to the bare binary otherwise):

> **Status: untested.** `slurmctld` starts, but `slurmd` cannot initialize
> because Slurm 23.11's cgroup/v2 plugin needs systemd/D-Bus scopes that this
> container lacks. `sbatch --test-only` fails with `Requested node
> configuration is not available`; no job was executed.

```sh
bash deploy/slurm/submit.sbatch            # interactive submission
sbatch deploy/slurm/submit.sbatch          # single run
bash deploy/slurm/sweep.sh                 # job array over the shape sweep
```

## 3. Kubernetes

> **Status: untested.** No `kubectl`, k3s, or microk8s is available, and the
> container runtime cannot create containers. The device plugin, GPU
> allocation, probes, and Job were not validated.

See `deploy/k8s/README.md` for the full walkthrough:

```sh
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.15.0/deployments/static/nvidia-device-plugin.yml
kubectl describe node <gpu-node> | grep nvidia.com/gpu
kubectl apply -k deploy/k8s
kubectl logs job/gpu-benchmark
```

## Scheduler daemon flags

```
--jobs N            synthetic jobs (0 = serve until SIGINT)
--arrival-rate R    poisson arrivals in jobs/sec
--saturation        submit all jobs at once for a peak-throughput run
--replay <csv>      replay jobs from CSV
--max-batch N       jobs per packed batch
--workers N         worker threads / streams
--impl tiled|cublas GEMM kernel implementation
--margin-mb MB      GPU memory safety margin
--http-port P       status server port (0 disables), default 8080
--force-cpu         run without a GPU (batches are dropped; for kind/minikube)
```

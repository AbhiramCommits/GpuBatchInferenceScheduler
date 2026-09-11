# Deployment: the identical workload, three ways

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
| 1. Bare host binary | workstation / Slurm compute node | `./build/bench_gemm` | stdout (tee to a file) |
| 2. Container | workstation (docker) or Slurm via srun + Pyxis/enroot | `docker run --gpus all ...` / `srun --container-image=...` | stdout, or `results/*.csv` |
| 3. Kubernetes | GPU cluster (`nvidia.com/gpu: 1`) | `kubectl apply -k deploy/k8s` | Job logs + `benchmark-results` PVC |

The long-running scheduler daemon (`scheduler_daemon`) ships in the same
image; it is deployed by `deploy/k8s/deployment.yaml` and can be probed on
`GET /healthz` and `GET /metrics`.

## Build the image

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

```sh
bash deploy/slurm/submit.sbatch            # interactive submission
sbatch deploy/slurm/submit.sbatch          # single run
bash deploy/slurm/sweep.sh                 # job array over the shape sweep
```

## 3. Kubernetes

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
--replay <csv>      replay jobs from CSV
--max-batch N       jobs per packed batch
--workers N         worker threads / streams
--impl tiled|cublas GEMM kernel implementation
--margin-mb MB      GPU memory safety margin
--http-port P       status server port (0 disables), default 8080
--force-cpu         run without a GPU (batches are dropped; for kind/minikube)
```

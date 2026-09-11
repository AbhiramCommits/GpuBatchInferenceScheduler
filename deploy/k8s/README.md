# Kubernetes deployment

The manifests deploy the scheduler daemon (a Deployment with probes and a
ConfigMap), a ClusterIP Service, and a one-shot GEMM benchmark Job that
writes its CSV to a PersistentVolumeClaim. The daemon exposes:

- `GET /healthz` — used by the liveness and readiness probes
- `GET /metrics` — Prometheus text format: queue depth, in-flight jobs,
  submitted/completed/failed totals, p95 latency, GPU utilization

## Prerequisites: NVIDIA device plugin

```sh
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.15.0/deployments/static/nvidia-device-plugin.yml
```

Verify the GPU capacity is visible on the node (the scheduler uses the
`nvidia.com/gpu.present` node label the plugin adds):

```sh
kubectl describe node <gpu-node> | grep nvidia.com/gpu
# nvidia.com/gpu:     1
# nvidia.com/gpu.present: true
```

## Deploy

```sh
kubectl apply -k deploy/k8s
```

This creates the ConfigMap, Service, Deployment, PVC and the one-shot
benchmark Job. Check status:

```sh
kubectl get pods
kubectl logs deployment/gpu-scheduler -f
```

The benchmark Job writes to the `benchmark-results` PVC:

```sh
kubectl wait --for=condition=complete job/gpu-benchmark --timeout=300s
kubectl logs job/gpu-benchmark
kubectl delete job gpu-benchmark   # remove after inspection
```

Query the status endpoints from inside the cluster:

```sh
kubectl run curl --rm -i --tty --image=curlimages/curl -- sh
curl http://gpu-scheduler:8080/healthz
curl http://gpu-scheduler:8080/metrics
```

Or from your workstation via port-forward:

```sh
kubectl port-forward svc/gpu-scheduler 8080:8080
curl http://localhost:8080/healthz
curl http://localhost:8080/metrics
```

Tune the scheduler by editing the ConfigMap (`JOBS`, `ARRIVAL_RATE`,
`WORKERS`, `MAX_BATCH`, `IMPL`, `MARGIN_MB`, `HTTP_PORT`) and rolling the
Deployment:

```sh
kubectl edit configmap scheduler-config
kubectl rollout restart deployment gpu-scheduler
```

## Testing locally without a GPU (minikube / kind)

No GPU is required to exercise the manifests and the HTTP endpoints:

1. Build the image (see `deploy/README.md`) and load it into the cluster:
   `minikube image load gpuinfer/scheduler:latest` or
   `kind load docker-image gpuinfer/scheduler:latest`.
2. The Deployment's `nodeSelector` will not match (no GPU label), so remove
   it and add `--force-cpu`:

```sh
kubectl apply -f deploy/k8s/configmap.yaml -f deploy/k8s/service.yaml
kubectl apply -f deploy/k8s/deployment.yaml
kubectl patch deployment gpu-scheduler --type=json \
  -p='[{"op":"remove","path":"/spec/template/spec/nodeSelector"}]'
kubectl patch deployment gpu-scheduler --type=json \
  -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--force-cpu"}]'
```

3. Wait for the pod to become Ready (the daemon drops jobs without a GPU but
   still serves the probes), then port-forward and curl `/healthz` and
   `/metrics` as above.

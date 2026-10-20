---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: omlx-dynamo-uninstall
description: Removes the Dynamo side of the oMLX-on-Mac example from Kubernetes. Takes one Mac out of the deployment or revokes an API token, deletes the DynamoGraphDeployment with its Ingress, token check, stream repair, Secrets and benchmark, and on request the dynamo-platform operator with its CRDs, the Prometheus and Grafana monitoring with its data, the images on the nodes, and the Linux host's key, token and client files. Use when taking a Mac out of service, decommissioning the cluster side, or starting over before omlx-dynamo-install; omlx-mac-uninstall then cleans up the Macs.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - omlx
    - kubernetes
    - uninstall
---

# Skill: Uninstall the oMLX Example From Kubernetes

The deployment lives in the namespace `dynamo-omlx`, the operator in `dynamo-system` and the
monitoring in `monitoring`. Paths below are relative to `examples/backends/omlx`.

## Ground Rules

- Agree the scope with the operator before removing anything. Each step below is a decision of
  its own: the platform, the monitoring and the CRDs may serve other workloads, and CRDs are
  cluster-wide.
- Keep the order: the deployment before the platform, the platform before its CRDs. Without
  the operator, the DGD's finalizer holds up its own deletion and the namespace's.
- Never print the Macs' API key or an API token, and never read a Secret with `-o yaml`,
  `-o json` or `describe`.
- Mac steps belong to `omlx-mac-uninstall` and run on the Mac.

## 1. Take One Mac Out

Take the Mac's component (`deploy/components/mac-<n>`) out of the overlay's `components`, and
renumber the components of later Macs to close the gap. Then:

```bash
R=$(mktemp)
kubectl kustomize --load-restrictor=LoadRestrictionsNone deploy/overlays/<overlay> > "$R"
kubectl apply --server-side -f "$R"
kubectl -n dynamo-omlx delete secret omlx-mac-<n>
```

The operator removes that Mac's proxy, and the router stops sending it requests. `mac-1` is in
the base, so removing it means removing the deployment (step 3). Continue on the Mac with
`omlx-mac-uninstall`.

## 2. Revoke an API Token

Delete the holder's file in `~/.config/dynamo-omlx/api-tokens/`, then rewrite the Secret from
the files that are left:

```bash
TOKENS=~/.config/dynamo-omlx/api-tokens
kubectl -n dynamo-omlx create secret generic omlx-api-tokens --from-file="$TOKENS" \
    --dry-run=client -o yaml | kubectl apply --server-side -f -
kubectl -n dynamo-omlx rollout restart deploy/omlx-auth    # refuse it at once
```

Without the restart the token check refuses the token within two minutes. A client that no
longer has a token gets 401.

## 3. Remove the Deployment

```bash
R=$(mktemp)
kubectl kustomize --load-restrictor=LoadRestrictionsNone deploy/overlays/<overlay> > "$R"
kubectl delete -f "$R" --ignore-not-found
kubectl -n dynamo-omlx delete job omlx-aiperf --ignore-not-found
kubectl -n dynamo-omlx wait --for=delete dgd/omlx --timeout=5m
kubectl delete namespace dynamo-omlx
```

- Deleting the rendered overlay removes the DGD, whose frontend and proxies the operator then
  removes, and the Ingress with its Middleware, the token check, the stream repair and the
  current ConfigMaps. Traefik answers `/v1` with 404 from then on.
- Deleting the namespace takes the rest: the Secrets (`omlx-mac-<n>`, `omlx-api-tokens`,
  `hf-token-secret`), the ConfigMaps of earlier rollouts and the benchmark, and the workers'
  `dynamoworkermetadatas`.
- The Macs keep serving oMLX until `omlx-mac-uninstall` stops them.

## 4. Remove the Platform

Only once no other DynamoGraphDeployment needs the operator:

```bash
kubectl get dynamographdeployments -A               # nothing besides what you remove
helm uninstall dynamo-platform -n dynamo-system --wait
kubectl delete namespace dynamo-system
```

Helm removes the operator, its two admission webhooks and its cluster roles and bindings. The
Dynamo CRDs stay, because the chart keeps them (`helm.sh/resource-policy: keep`). Deleting them
deletes every object of those kinds in the cluster, so first check that nothing lists:

```bash
kubectl get dgd,dcd,dgdr,dgdsa,dm,dwm -A
kubectl get crd -o name | grep -E '/dynamo[a-z]*\.nvidia\.com$'    # the six Dynamo CRDs
kubectl get crd -o name | grep -E '/dynamo[a-z]*\.nvidia\.com$' | xargs kubectl delete
```

The pattern takes only Dynamo's own CRDs. Other `nvidia.com` CRDs, such as the GPU Operator's,
stay.

## 5. Remove Monitoring

```bash
kubectl delete -f deploy/monitoring/grafana-omlx-dashboard.yaml \
    -f ../../../deploy/observability/grafana-dynamo-dashboard-configmap.yaml --ignore-not-found
```

That alone removes the example's two dashboards from Grafana. To remove the whole stack, when
nothing else uses it:

```bash
helm uninstall prometheus -n monitoring --wait
kubectl delete namespace monitoring
kubectl -n kube-system delete service prometheus-kube-prometheus-kubelet --ignore-not-found
```

- Helm leaves three things behind:
  - Prometheus's 20 GiB volume with the metrics history, which goes with the namespace.
  - The kubelet Service that prometheus-operator created in `kube-system`, which the last
    command deletes.
  - The ten `monitoring.coreos.com` CRDs
    (`kubectl get crd -o name | grep '\.monitoring\.coreos\.com$'`). Delete them only when no
    other chart ships ServiceMonitors, PodMonitors or PrometheusRules: deleting them deletes
    every such object in the cluster.

## 6. Remove the Images

On each node, once no other workload uses them:

```bash
sudo k3s ctr -n k8s.io images rm nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.5.1 \
    nvcr.io/nvidia/ai-dynamo/kubernetes-operator:1.5.1
```

The frontend image alone is 3.9 GB.

## 7. Clean Up the Linux Host

Only for what nothing else uses:

- `~/.config/dynamo-omlx/api_key`, the Linux copy of the Macs' key, once no Mac and no
  deployment uses that key. The Macs keep theirs until `omlx-mac-uninstall`.
- `~/.config/dynamo-omlx/api-tokens/`, the clients' tokens. Once the Secret is gone they open
  nothing.
- The clients' settings that point at the deployment, for example Claude Code's configuration
  directory for it and Codex's `~/.codex/omlx.config.toml`.
- Saved AIPerf logs and the CSVs split from them.
- Whatever published the API (step 8 of `omlx-dynamo-install`): a front proxy route, a tunnel or
  a DNS record still points at the node and now serves 404.
- `~/.config/dynamo-omlx/` itself, once it is empty.

## 8. Verify

```bash
kubectl get dgd -A                                  # no omlx
kubectl get namespace dynamo-omlx                   # NotFound
helm list -A | grep -E 'dynamo-platform|prometheus' # only the releases you kept
curl -s -o /dev/null -w '%{http_code}\n' http://<node address>/v1/models     # 404
```

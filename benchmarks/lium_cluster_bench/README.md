<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# lium_cluster_bench: multi-node Dynamo + SGLang throughput harness on Lium.io

One command rents (or attaches to) a 2-, 4- or 8-node x 8-GPU cluster, deploys Dynamo +
SGLang serving GLM-5.3, searches for the configuration with the highest output
tok/s/GPU on the SemiAnalysis Claude Code trace (AIPerf 0.13.0, AgentX
closed-loop replay), measures interconnect speed-of-light and under-load
occupancy, compares 1/2/4/8 IB NICs per node and NVLink against PCIe
(the leader re-measured with NVLink switched off), reports the leader's three
focus resources (the minimum interconnect it needs, its model FLOPs
utilization, and how much context its KV cache holds), records the upstream switch
topology and the cluster's deployment-time state (node facts and health,
service metrics, GPU activity, bottleneck, energy, cold start, goodput, cost),
gates the recommendation on a GSM8K accuracy check (throughput runs use
`ignore_eos`, so broken numerics would look fast), optionally captures a
kernel-time breakdown, and writes a Chinese report plus an incrementally
mergeable knowledge base for later agentic tests.

Cluster sizes: profiles are written for 4 x 8 GPUs and fitted to the rented
size. Instance counts scale by the exact factor (`pd-{prefill}p{decode}d` is
`pd-2p1d` on 2 nodes, `pd-4p2d` on 4, `pd-8p4d` on 8) and so do concurrency
grids; `nodes:` limits a family to some sizes (2-node DTP4 P:D alternates,
opt-in wide-EP DEP16/DEP64). The controller refuses a size no tuned family fits
before renting (`--nodes 2|4|8`; other sizes keep the families that scale
exactly).

Supported GPU profiles: B300, B200, H200, H100 (2-node replicas), RTX PRO 6000
Blackwell (PCIe only; DSA kernels are verified by a dummy-weight smoke ladder).

## 快速开始

```bash
# 0. 依赖：本机有 uv（https://docs.astral.sh/uv/）和 lium CLI；凭据只从环境变量读取
export LIUM_API_KEY=...        # 仅在本机使用，不会发送到集群
export HF_TOKEN=...            # 经 SSH stdin 传给 node0，不落盘、不进命令行

# 1. 一键：租 2/4/8 节点（--nodes，默认 4）-> 同步代码 -> 后台运行 -> 轮询进度 -> 拉回结果 -> 合并知识库 -> 释放集群
#    （--fabric auto 按 --gpu 选择 lium fabric；-- 之后的参数原样传给 node0 上的 runner）
bash benchmarks/lium_cluster_bench/cluster_bench.sh controller run-all --exp-id h200-001 \
    --gpu h200 --fabric auto --kb-root ~/cluster_bench_kb -- --budget-hours 9.5
#    分步执行：controller up | sync | launch | watch | pull | down | status（--keep 保留集群）

# 2. 已有集群（在 node0 上执行；wg0 上的其余节点自动发现，--expect-nodes 给出节点数）
bash benchmarks/lium_cluster_bench/cluster_bench.sh run --exp-id h200-001 --nodes auto --expect-nodes 8 \
    --price-per-node-hour 38.4   # 可选：用于 $/百万 token；lium 租用时自动从 provider.json 读取

#    可选：--diagnose 诊断时段（Nsight Systems 的 kernel 耗时拆分 + SGLang MFU/专家均衡度，不计入正式结果）
#          --goodput-slo NAME:TTFT_S:ITL_MS 自定义 goodput 档位；--no-accuracy 关闭精度检查（不推荐）
#          --link-target-util 0.7 估算最低互联配置时，链路 p99 占用的上限（默认 70%）
#          --no-nvlink-ablation 不做“禁用 NVLink 改走 PCIe”的对比（默认做，最后会恢复 NVLink）

# 3. 中断后续跑：同一个 --exp-id 再执行一次即可（从事件日志精确恢复）
# 4. 不需要 GPU 的全流程演练（打印将执行的每一条命令，生成示例报告）
bash benchmarks/lium_cluster_bench/cluster_bench.sh dry-run --fake-gpu h200 --exp-id demo
```

Outputs per experiment (`<workdir>/experiments/<exp_id>/`):

| Path | Content |
|---|---|
| `report/report.zh-CN.md`, `report/report.zh-CN.html` | Concise Chinese report (single-file HTML with inline SVG charts) |
| `analysis/summary.json` | `ExperimentSummary` - the only input of report and KB writers |
| `events.jsonl` | Append-only, CRC-checked event log (drives resume) |
| `points/<id>/` | Sealed raw evidence per measured point (AIPerf exports, telemetry JSONL, SHA256SUMS) |
| `deployments/<fp>/` | Rendered `plan.json`, per-container configs/env files, container logs |
| `status.json` | Heartbeat polled by the laptop controller |

The knowledge base (`<workdir>/kb/`, merged into the laptop KB by the
controller) follows `agent-docs` conventions: per-hardware-fingerprint
records (`runs`, `peaks`, `util`, `nic_sweep`, `findings`, `fabric`, `health`,
`accuracy` JSONL, deduplicated by content hash) plus regenerated Markdown views
(`speed-of-light.md`, `topology-ranking.md`, `knob-findings.md`,
`fabric-and-health.md`, `resources.md`; `INDEX.md` lines up one hardware shape across 2/4/8
nodes; `topology-ranking.md` carries each best measurement's accuracy verdict
and goodput per SLO tier) and a run-artifacts style `experiments/<exp_id>/`
folder.

## How a run works

| Phase | What happens |
|---|---|
| preflight | Replicate code to peers, bootstrap venvs, probe GPUs/HCAs/topology/disks, pick profiles and a local weights disk (never `/mnt`), pull the image, verify every SGLang/Dynamo/AIPerf flag against `--help` of the pinned versions |
| download | Resolve the HF commit, download 1/N of the shards per node in the background, meanwhile run intra-node P2P microbenchmarks and a dummy-weight kernel smoke (walks the GPU profile's ladder), then NCCL-broadcast the missing shards over the RDMA fabric and verify sha256; then measure the weights disk's direct-read speed |
| fabric | Upstream switch topology and link health (see below): identify (sysfs / MAD / LLDP), traces between every node pair per rail, a perftest link matrix (round-robin pairs, every rail concurrently, ~2-4 min on 8 nodes; `--no-link-matrix` skips it), and the list of switch ports sampled at every load point |
| microbench | Inter-node NCCL collectives and GPU-direct send/recv with 8/4/2/1 NICs (speed-of-light references) |
| search | Topology families in priority order (recipe baseline first), powers-of-two concurrency climb per family, pruning, P:D alternate by symptom, floor bracketing, MTP-off ablation |
| accuracy | GSM8K 5-shot (200 questions, sha256-pinned test split, greedy, Dynamo `/v1/completions`) on the leader; a configuration below the floor (0.80) or more than 0.07 below the best checked one (outside the 95% CI) never leads, so the next best is checked (`--accuracy-max-checks`, default 3); `--no-accuracy` disables it |
| pilot | n=3 repetitions at the leader's knee: noise floor and minimum detectable effect |
| nic_sweep | 1/2/4/8 NICs on the best configuration that has measured cross-node traffic |
| confirm | Decision-length (>= 900 s, no `--unsafe-override`) rerun of the leader |
| nvlink | NVLink against PCIe: tear down, switch NVLink out of the GPU-GPU path on every node (Hopper: `nvidia-smi nvlink -sBwMode OFF`, "P2P is via PCIe bus"; Blackwell: `-sLWidth 0` only if offered), prove it with a GPU-GPU copy benchmark (must drop to PCIe speed), deploy the leader with NVLS and SGLang's custom all-reduce off (`NCCL_P2P_LEVEL=SYS`), measure one point at the leader's concurrency, restore NVLink (also on resume after a crash and in finalize). Without root or a supported switch: NCCL without P2P, staged through host memory (a labelled lower bound). A DeepEP leader (NVLink-only intranode kernels) hands the comparison to the best family without DeepEP; PCIe-only GPUs skip it. The run is invalid if NVLink counters show traffic. Its budget is the first to yield: the leader's climb and every earlier step may spend it (`--no-nvlink-ablation` turns it off) |
| diagnose | Opt-in (`--diagnose`): the leader redeployed as a diagnostic variant (SGLang `--enable-mfu-metrics`, device timer, `--expert-balancedness-report-mode prometheus` for MoE; workers under Nsight Systems from the pinned image or a derived `cb-diag/...` image built on each node), a short load, `POST /engine/control/start_profile` for 10 forward steps, `nsys stats` -> kernel-time shares per role. Never ranked |
| finalize | "after" health snapshot, summary, report, KB export |

## Focus resources of the leader

Report section 2 and the KB view `resources.md` answer these questions about
the leader at its saturation point (2.1 interconnect, 2.2 NVLink against
PCIe, 2.3 MFU, 2.4 KV capacity), each with the method stated next to the
number:

| Question | How it is derived | Where |
|---|---|---|
| How much interconnect does it need? | For each link kind (NVLink per GPU, GPU PCIe, GPU<->NIC PCIe, IB/RoCE per node) the busiest device's p99 over 1-s bins, worse direction; utilization against the nominal per-direction rate; need = p99 / `--link-target-util` (default 0.7). Fewest NICs per node twice: measured by the NIC sweep (fewest within the noise floor with every larger count also holding; the sweep runs on the best cross-node family when the leader has none, and is attributed to it) and estimated from the leader's own IB need / one NIC's rate. NVLink is needed when the GPU-GPU need exceeds one PCIe link | `summary.interconnect`, KB `index.jsonl` (`min_nics_*`, `ib_*`, `nvlink_*`) |
| Does it need NVLink? | Measured: the same configuration at the same concurrency with NVLink switched off (phase `nvlink` above): tok/s/GPU, per-user rate, TTFT and ITL against NVLink, the delta judged against the noise floor, the method, what carried the traffic (PCIe P2P or host staging), the GPU-GPU copy rate and PCIe p99 during the run | `summary.nvlink`, KB `index.jsonl` `nvlink_off_*`, `run.jsonl` `p2p` |
| How much of the GPUs' floating-point peak does it use (MFU)? | Model FLOPs from the checkpoint's `config.json` (MLA, MoE with shared experts and dense layers, DSA top-k attention plus its indexer, dense GQA) for useful tokens only: prompt tokens not served from the prefix cache (per-request `usage` or the server's hit rate) and generated tokens; MTP drafts are excluded. Divided by window x GPUs x the GPU profile's dense peak at the checkpoint's GEMM precision; per role for PD; NVML GPM tensor activity alongside | `PointResult.mfu`, `summary.model_arch`, KB `run.jsonl` `mfu_*`, `index.jsonl` `leader_mfu_*` |
| How much context fits in the KV cache? | Per-rank pool `max_total_num_tokens` (cold-start log) x attention-DP ranks x instances of the serving role (agg, or decode with PD); longest single request = one rank's pool capped by the context length; concurrent requests at the trace's p50/p90 context (ISL + OSL); pool usage at the leader point; bytes per token measured vs from `config.json`; HiCache host tokens | `summary.kv_capacity`, KB `index.jsonl` `kv_*` |

## What is collected for the knowledge base

Every collector is fail-soft: a missing tool, permission or device lowers
coverage (and says so in the report) but never stops the benchmark. Facts
carry the tier that produced them: S sysfs, M MAD/SMP (needs
`/dev/infiniband/umad*` and an SMP-permitting fabric), L LLDP, T unprivileged
UDP traceroute, P perftest / perfquery, D derived.

| Area | Data | When | Where |
|---|---|---|---|
| Switch topology | per HCA port: LID/SM/GID subnet, rate, MTU, RoCE GID/IP, PCIe link; leaf switch identity and port (SMP directed route), leaf port census (uplinks vs hosts, other tenants' hosts, oversubscription), spines, SM; RoCE: LLDP chassis/port/system name, PFC/ETS/APP TLVs, `mlnx_qos`; hop counts per rail (ibtracert / UDP traceroute); derived: rail-optimized, nodes per leaf, tiers, degraded links | fabric phase | `fabric/`, `summary.fabric*`, report section 9, KB `fabric.jsonl` |
| Link health | perftest RDMA write bandwidth and latency for every node pair x rail (weak links, asymmetry, latency groups = same leaf vs via spine) | fabric phase | `summary.fabric_links`, `link_findings` |
| Switch load under load | PortXmitData/RcvData/XmitWait/discards/errors of every leaf port facing our HCAs and every leaf uplink, before/after each AIPerf run (PMA queries from node0) | every point | `points/<id>/switch_load.json`, `PointResult.switch_load` |
| NIC / GPU stress | IB congestion and error counters (port_xmit_wait, discards, CNP/ECN, out_of_buffer, ...), RoCE PFC/pause counters (`ethtool -S`), GPU clocks, temperature, power, energy, throttle reasons | every point (telemetry, 5 s slow tick) | `points/<id>/stress.json`, `PointResult.state.stress` |
| Service state | Dynamo frontend / router and SGLang worker `/metrics` scraped by AIPerf (names checked against the SGLang 0.5.18 source): running/queued requests, KV usage, prefix hit rate, PD queues, KV transfer (latency, bootstrap, allocation wait), per-request stage latencies (`per_stage_req_latency_seconds`: bootstrap, prefill, KV transfer, decode wait), CUDA graph coverage of decode passes, retracted and evicted tokens, request-weighted MTP acceptance (generated tokens per verify step) and acceptance rate; bottleneck label with evidence; energy per output token | every point | `points/<id>/servicestate.json`, `PointResult.state` |
| GPU activity | NVML GPM per GPU at 1 s: SM, occupancy, tensor, DRAM bandwidth, PCIe and NVLink rates; regime per role (compute / memory / waiting / idle) for the bottleneck label; per-GPU availability (`unsupported`, `NVML_ERROR_*`) is data | every point (telemetry agent; off while nsys runs) | `stress.json` `activity`, `PointState.stress.activity` |
| Requests | From AIPerf's per-request records (`--export-level records`, no extra load): goodput under SLO tiers (`u50`: TTFT <= 10 s and ITL <= 20 ms; `u20`: 30 s / 50 ms; `--goodput-slo`), the best point per tier, TTFT by session turn (prefix reuse), session spans and subagent requests | every point | `points/<id>/requests.json`, `PointResult.requests` |
| Accuracy | GSM8K result, Wilson 95% CI, verdict and reason per checked configuration; answer source (text / reasoning field) | accuracy phase | `accuracy/`, `summary.accuracy`, KB `accuracy.jsonl` |
| Diagnostics (opt-in) | kernel-time shares by category (attention, MoE GEMM, dense GEMM, DeepEP, NCCL, KV cache, ...) and top kernels per role, per-GPU trace coverage, nsys version; SGLang estimated TFLOP/s and memory traffic per GPU, forward-busy share, EPLB balancedness | diagnose phase | `diagnostics/`, `summary.diagnostics`, KB `index.jsonl` `diag_*` |
| Cold start | per container: time to weights loaded, KV pool, CUDA graphs, ready; weight/KV memory, `max_total_num_tokens`, `max_running_requests`; NCCL/OOM/RDMA log issues | every deployment | `deployments/<fp>/lifecycle.json`, `DeployResult.lifecycle` |
| Node facts and health | PCIe links (incl. upstream bridges, AER/ACS), NUMA, hugepages/THP, `/dev/shm`, memlock, cgroup, IOMMU, kernel modules, driver flavour, `nvidia-smi -q -x` (ECC, row remap, power limits, clocks, fabric/clique), NVLink status/errors, P2P matrices, HCA counters, kernel log (Xid, mlx5) when readable; "before" and "after" snapshots -> what grew under load | preflight, finalize | `facts/`, `summary.fact_findings`, `health_deltas`, report section 11, KB `health.jsonl` |
| Environment | software stack inside the pinned image (sglang, flashinfer, torch/NCCL, nixl, dynamo, UCX, OS), clock offsets between nodes, weights-disk read speed | preflight, download | `summary.software`, `clock_offsets`, `disk_reads` |
| Cost | lium fabric id/type/link rate and price (controller `provider.json`, no secrets or IPs) or `--price-per-node-hour`: cluster $/h, experiment cost, $/1M output tokens at saturation and at the interactivity floors | finalize | `summary.provider`, `summary.cost`, KB `index.jsonl` |

Node bootstrap installs `infiniband-diags ibverbs-utils perftest` with apt when
they are missing on a pod (best effort, once; `CB_FABRIC_TOOLS=0` disables it),
so MAD queries and the link matrix work even if the image lacks them.

Key facts the harness relies on (verified in the AIPerf 0.13.0 wheel): every
AIPerf run cache-busts first-turn prefixes with a fresh uuid, so concurrency
points need no worker restart; `--max-context-length` drops whole traces, so
Hopper profiles serve 262144 tokens instead of the recipe's 250000.

## Layout

```text
domain/    pure models and logic, no I/O (schema, profiles, inventory, nicplan, search, analysis, bandwidth,
           fabric, nodefacts, servicestate, loadstress, lifecycle, pointstate, switchload, cost, requests,
           gpuactivity, accuracy, kernels, modelarch (config.json FLOPs/KV bytes, MFU), capacity
           (interconnect needs, fewest NICs, KV capacity), nvlink (switch methods, NVLink-off comparison))
ports.py   protocols the application depends on (Executor, Cluster, Provider, Engine, TransportPlugin, Clock)
fsutil.py  crash-safe file primitives shared by every layer (atomic writes, file lock, sha256)
adapters/  profile_store (YAML profiles + recipe ConfigMaps), ssh/docker executors, lium + existing-cluster
           providers, SGLang renderer, transports, AIPerf, HF fetch, telemetry, microbenchmarks,
           fabric prober, node-facts collector, accuracy probe runner, Nsight Systems diagnostics
agents/    standalone node-side scripts (probe, telemetry sampler with NVML GPM, P2P/NCCL benchmarks,
           NCCL file gather, fabric_probe, node_facts, accuracy_probe)
app/       event store, deployment session, pipeline and phases (incl. accuracy gate, diagnose), fabric/facts
           collection points, model_ops (config.json read once per run, MFU per point), nvlink_ops
           (switch NVLink off on every node and back), summary, simulator
report/    Chinese Markdown/HTML report      kb/  knowledge-base records and writer
profiles/  gpus/*.yaml, models/*.yaml, families/<model>/<gpu>.yaml (recipe ConfigMaps + overrides + ladder)
```

Adding a GPU type or model is YAML only. A new transport is one module in
`adapters/transports/` registered in `REGISTRY`, plus its name in the
`Transport` literal of `domain/schema.py`. A new cluster source implements
`ports.Provider`.

`summary.json` and the knowledge base keep English text; the Chinese report
localizes known reason templates at render time (`report/reasons.py`).

## Tests

```bash
python -m pytest --confcutdir=tests/benchmarks tests/benchmarks/lium_cluster_bench
```

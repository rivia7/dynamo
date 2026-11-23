<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# cluster_bench: multi-node Dynamo + SGLang throughput harness

One command takes a GPU cluster (machines you reach over SSH, or a Lium.io
rental), deploys Dynamo + SGLang serving GLM-5.3 (or any `--model-repo`), and
finds the configuration with the highest throughput per GPU on a chosen
dataset (AIPerf 0.14.0; by default AIPerf's AgentX short command,
`--scenario agentx --public-dataset semianalysis_cc_traces_weka_062126`). It
measures one of three stages:

| `--stage` | What every GPU runs | Objective |
|---|---|---|
| `e2e` (default) | the profile's aggregated and PD-disaggregated families, end to end | output tok/s/GPU |
| `p-only` | the prefill role of the PD families, one output token per request | input (prompt) tok/s/GPU |
| `d-only` | the decode role of the PD families, KV transfer faked | output tok/s/GPU |

P-Only and D-Only answer "how much prefill, or decode, can this cluster do at
most on this dataset and model"; the knowledge base turns the latest P-Only and
D-Only leaders of one cluster into the throughput-ideal P:D GPU split (see
[Stages](#stages-e2e-p-only-d-only)).

Along the way it measures interconnect speed-of-light and occupancy under
load, compares 1/2/4/8 IB NICs per node and NVLink against PCIe (the leader
re-measured with NVLink switched off), reports the leader's three focus
resources (the minimum interconnect it needs, its model FLOPs utilization, and
how much context its KV cache holds), records the upstream switch topology and
the cluster's deployment-time state (node facts and health, service metrics,
GPU activity, bottleneck, energy, cold start, goodput, cost), gates E2E
recommendations on a GSM8K accuracy check (throughput runs use `ignore_eos`,
so broken numerics would look fast), optionally captures a kernel-time
breakdown, and writes a Chinese report plus an incrementally mergeable
knowledge base for later agentic tests.

Cluster sizes: profiles are written for 4 x 8 GPUs and fitted to the cluster.
Instance counts scale by the exact factor (`pd-{prefill}p{decode}d` is
`pd-2p1d` on 2 nodes, `pd-4p2d` on 4, `pd-8p4d` on 8) and so do concurrency
grids; `nodes:` limits a family to some sizes (2-node DTP4 P:D alternates,
opt-in wide-EP DEP16/DEP64). P-Only and D-Only families fill every GPU with
one stage role, so they also run on a single node when one instance fits in
it. Before renting, the Lium controller refuses a size that no tuned family
fits (`--nodes 2|4|8`; other sizes keep the families that scale exactly).

Supported GPU profiles: B300, B200, H200, H100 (2-node replicas), RTX PRO 6000
Blackwell (PCIe only; DSA kernels are verified by a dummy-weight smoke ladder)
and RTX PRO 6000D (PCIe only, 89.8 GB: attention TP8 instead of DP8, because
the replicated BF16 attention weights would not fit).
Other GPUs and models run untuned with `--model-repo` and `--model-size-gb`.

GLM-5.3-Flash (`--model glm-5.3-flash`: hybrid KDA linear attention plus DSA
MLA layers) is tuned for the RTX PRO 6000D only, as TP4 and TP8 roles. On SM120
its attention runs only on SGLang's TileLang DSA kernels with a BF16 KV cache,
and SGLang 0.5.21 needs one source edit for that. Such edits are named in
`agents/sglang_hotfix.py`; a role lists the ones it needs in its
`CB_SGLANG_HOTFIXES` env, `scripts/launch_node.sh` applies them inside the
worker container before the engine starts, and preflight checks that each one
fits the pinned image (`preflight/sglang_hotfixes.txt`), so a run never serves
an unpatched engine. The profile serves 229376 tokens, as GLM-5.3 does on that
GPU, so both models replay the same AgentX traces.

## 快速开始

```bash
# 0. 依赖：本机有 uv（https://docs.astral.sh/uv/）和 ssh；只有租用 Lium 时才需要 lium CLI。凭据只从环境变量读取
export HF_TOKEN=...            # 经 SSH stdin 传给 node0，不落盘、不进命令行
export LIUM_API_KEY=...        # 仅 Lium：只在本机使用，不会发送到集群

# 1. 自己的机器：每行一台机器的 SSH 访问方式，第一行是 node0（harness 在 node0 上运行）
cat > machines.txt <<'EOF'
ssh -p 2222 -i ~/.ssh/lab root@203.0.113.10    control_ip=10.0.0.11
root@203.0.113.11:2222                           name=gpu-b
ssh -J admin@bastion.example.com root@10.0.0.13
EOF
#    一键：检查机器 -> 同步代码 -> 后台运行 -> 轮询进度 -> 拉回结果 -> 合并知识库 -> 清理（只清理本次实验的容器和密钥，机器保留）
#    （-- 之后的参数原样传给 node0 上的 runner）
bash benchmarks/cluster_bench/cluster_bench.sh controller run-all --exp-id h200-ponly-001 \
    --machines machines.txt --kb-root ~/cluster_bench_kb -- --stage p-only --budget-hours 6
bash benchmarks/cluster_bench/cluster_bench.sh controller run-all --exp-id h200-donly-001 \
    --machines machines.txt --kb-root ~/cluster_bench_kb -- --stage d-only --budget-hours 6
#    同一集群、同一模型和数据集的 P-Only 与 D-Only 都测完后，KB 的 INDEX.md 给出 PD 配比
#    分步执行：controller up | sync | launch | watch | pull | down | status
#    也可以不写文件：--machine 'ssh -p 2222 root@203.0.113.10' --machine root@203.0.113.11（可重复）
#    网络受限（连不上 huggingface.co / GitHub）：先把权重放到每台机器的同一路径，再加
#    -- --model-path /data/models/GLM-5.3-NVFP4 --checkpoint nvfp4-nvidia --hf-endpoint https://hf-mirror.com
#    （见下文 Restricted networks and pre-placed weights）

# 2. 指定数据集（默认 semianalysis_cc_traces_weka_062126：SemiAnalysis Claude Code trace 完整语料，
#    用 AIPerf 的 --scenario agentx 闭环回放）
#    -- --dataset sharegpt                         AIPerf 公开数据集
#    -- --dataset synthetic:isl=8192,osl=1024      合成请求
#    本机的 trace 文件：--upload 先传到 node0 的 <workdir>/inputs/，再按路径引用
bash benchmarks/cluster_bench/cluster_bench.sh controller run-all --exp-id h200-donly-002 \
    --machines machines.txt --upload ./my_trace.jsonl \
    -- --stage d-only --dataset mooncake_trace:/root/cluster_bench/inputs/my_trace.jsonl
#    其他 AIPerf 参数：-- --aiperf-arg=--goodput --aiperf-arg=time_to_first_token:2000
#    （负载、数据集、输出长度等由 harness 管理，不能覆盖）

# 3. Lium.io：租 2/4/8 节点（--nodes，默认 4），结束后释放（只释放 harness 自己租的集群）
#    （--fabric auto 按 --gpu 选择 lium fabric）
bash benchmarks/cluster_bench/cluster_bench.sh controller run-all --exp-id h200-001 \
    --gpu h200 --fabric auto --kb-root ~/cluster_bench_kb -- --budget-hours 9.5

# 4. 直接在 node0 上运行（不经过本机控制器）
#    自己的机器：hosts 文件每行 name address [ssh_target] [ssh_port] [identity=PATH] [iface=NIC] [known_hosts=PATH]，
#    第一行是 node0（ssh_target 写 - 表示本机）；address 是节点的私网控制地址
bash benchmarks/cluster_bench/cluster_bench.sh run --exp-id h200-001 --hosts hosts.txt --stage p-only
#    Lium：wg0 上的其余节点自动发现，--expect-nodes 给出节点数
bash benchmarks/cluster_bench/cluster_bench.sh run --exp-id h200-001 --platform lium \
    --nodes auto --expect-nodes 8 \
    --price-per-node-hour 38.4   # 可选：用于 $/百万 token；lium 租用时自动从 provider.json 读取

#    可选：--diagnose 诊断时段（Nsight Systems 的 kernel 耗时拆分 + SGLang MFU/专家均衡度，不计入正式结果）
#          --goodput-slo NAME:TTFT_S:ITL_MS 自定义 goodput 档位；--no-accuracy 关闭精度检查（不推荐）
#          --link-target-util 0.7 估算最低互联配置时，链路 p99 占用的上限（默认 70%）
#          --no-nvlink-ablation 不做“禁用 NVLink 改走 PCIe”的对比（默认做，最后会恢复 NVLink）
#          --confirm-duration-s 3600 长时复测按 agentx 预设的 3600 s 跑（默认 1200 s；AgentX 要求 ≥ 900 s）
#          --image nvcr.io/nvidia/ai-dynamo/sglang-runtime-nightly:<YYYYMMDD-sha> 换成 nightly 镜像（SGLang 0.5.23）

# 5. 中断后续跑：同一个 --exp-id 再执行一次即可（从事件日志精确恢复）
# 6. 不需要 GPU 的全流程演练（打印将执行的每一条命令，生成示例报告）
bash benchmarks/cluster_bench/cluster_bench.sh dry-run --fake-gpu h200 --fake-nodes 2 --stage p-only --exp-id demo
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
nodes and turns P-Only + D-Only leaders into a PD split; `topology-ranking.md` carries each best measurement's accuracy verdict
and goodput per SLO tier) and a run-artifacts style `experiments/<exp_id>/`
folder.

## SSH machines

`controller up --machines FILE` (or repeated `--machine 'ssh ...'`; `run-all`
takes the same flags) records the machines you bring. Nothing is rented, and
`down` never removes a machine.

The machines file lists one machine per line, node0 first; `#` starts a
comment. A line is how your laptop reaches the machine, then optional
`key=value` options:

| Part | Forms |
|---|---|
| Access | an `ssh` command (only `-p`, `-i`, `-l`, `-J` and `-o` with `Port`, `User`, `IdentityFile`, `ProxyJump`, `IdentitiesOnly`, `ConnectTimeout`, `ServerAliveInterval`, `ServerAliveCountMax`: nothing that runs a local command, such as `ProxyCommand`, and no remote command), `user@host[:port]`, or `ssh://user@host[:port]`. The user defaults to root |
| `name=LABEL` | label in `controller status` (default: the hostname) |
| `control_ip=IPv4` | the private address the cluster talks over (default: chosen as below) |
| `peer_ssh=[user@]host[:port]` | how node0 reaches this machine, as the same user (default: `user@<control_ip>` port 22, then your own target and port) |
| `iface=NIC` | the control-plane NIC (default: the one that holds the control address) |

What `up` does, in order:

1. Runs a discovery script on every machine over your access:
   hostname, uid, IPv4 addresses, `nvidia-smi` GPUs, whether docker works for
   that user, whether `--workdir` (default `/root/cluster_bench`) is
   writable (it creates the directory if missing). It reports every problem
   at once and changes nothing else until all machines pass. GPU model and
   count must match on every machine.
2. Picks each machine's control address (etcd/NATS, NCCL bootstrap, the
   probe's control NIC): `control_ip=` if given, else the first node0 subnet
   that every machine has an address on (private before public; `wg*`,
   `bond*`, `eth*`, `en*`, `ib*` first; never loopback, docker or CNI bridges).
   `iface=` limits a machine to that NIC.
3. Records the state in `~/.cluster_bench/<exp_id>/controller.json` before it
   changes anything.
4. Creates one ed25519 key for the experiment on node0
   (`<workdir>/ssh/<exp_id>_ed25519`), appends its public key to the ssh
   user's `~/.ssh/authorized_keys` on every other machine (comment
   `cluster_bench:<exp_id>`), and checks each node0 -> machine route by
   hostname. node0 records each machine's host key in
   `<workdir>/ssh/<exp_id>_known_hosts` and the runner checks against that
   file, so a stale `~/.ssh/known_hosts` entry (a reinstalled machine) does
   not break the run.

`launch` writes the runner's hosts file
(`<workdir>/experiments/<exp_id>/hosts.txt`) from the checked routes and starts
the runner with `--platform generic --hosts ...`. `down`, after a successful
`pull` (`--force` without one), removes this experiment's containers (docker
label `cb.exp`), the trust key and the known_hosts file from every machine,
node0 last;
`--keep` leaves them.

Each machine needs:

- Linux with the NVIDIA driver (`nvidia-smi`), and Docker with the NVIDIA
  container toolkit that the ssh user can run (root, or the `docker` group).
- An ACTIVE RDMA HCA (InfiniBand or RoCE): preflight requires one on every
  node, because the fabric, microbenchmark and NIC phases and the PD
  transports use it.
- Internet access on every machine: bootstrap installs uv, Python 3.12 and the
  harness's Python packages into `<workdir>/venv` (AIPerf 0.14.0 on node0),
  and every node pulls the image and downloads its share of the weights from
  Hugging Face. Behind a restricted network, see
  [Restricted networks and pre-placed weights](#restricted-networks-and-pre-placed-weights).
- Root works best: switching NVLink off for the NVLink/PCIe comparison needs
  it. With another user the controller warns; pass `-- --no-nvlink-ablation`
  if the comparison fails.

The controller puts user and port on the ssh command line, so write them in
the machines line: an `~/.ssh/config` alias still supplies `HostName`,
`IdentityFile` and `ProxyJump`, but not `User` or `Port`. A machine without
`-i` uses `--ssh-identity`, else ssh's own defaults and agent (never lium's
key).

### Restricted networks and pre-placed weights

Where huggingface.co, GitHub or PyPI is slow or blocked, prepare the machines
once and give the runner (after `--`) up to three flags:

- `--model-path /data/models/GLM-5.3-NVFP4`: the weights are already at this
  absolute path, the same on every node (copy them over the LAN). Nothing is
  downloaded and the space check is skipped. The download phase checks that
  every node holds the same files with the same sizes as node0, that
  `config.json` and every shard the safetensors index names are there, and
  that no `.aria2`, `.incomplete` or `.part` file is left; it does not hash
  them. Pin `--model-revision <commit>` to record which commit they are.
- `--checkpoint VARIANT`: which checkpoint of the tuned model profile the
  weights are, for example `nvfp4-nvidia` for `nvidia/GLM-5.3-NVFP4` (SGLang
  loads its mixed NVFP4/FP8 layers as `modelopt_mixed`). The checkpoint's
  `sglang:` keys are merged into every role, over the recipe and the family.
  Without the flag the GPU profile's `weights_variant` is used.
- `--hf-endpoint https://hf-mirror.com`: `HF_ENDPOINT` for what the nodes
  still fetch from Hugging Face: the AIPerf dataset, and the weights when
  there is no `--model-path`.

Bootstrap finds uv in `~/.local/bin` (for example `pip install --user uv`
from a PyPI mirror), and uv reads `~/.config/uv/uv.toml`, so mirrors there
cover the venv: a PyPI index and, for a host without Python 3.12, a
python-build-standalone mirror (top-level keys go before `[[index]]`):

```toml
python-install-mirror = "https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone"

[[index]]
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true
```

The pinned image comes from `nvcr.io`; pulling it on every machine before the
run saves the first deployment the wait.

## Lium.io specifics

Lium is a platform special case (`--platform lium`, which the controller sets
when it rents or adopts a cluster):

- The control plane is WireGuard `wg0` (10.42.0.x); `--nodes auto` finds the
  other nodes on it (`--expect-nodes N`).
- Model weights never go to `/mnt` (Lium's network volume). On other
  platforms `/mnt` is an ordinary candidate disk.
- `controller up` rents (`--gpu`, `--nodes 2|4|8`, `--fabric auto`,
  `--ttl-hours`) or adopts (`--adopt NAME`) a cluster, refusing a size no
  tuned family fits before it bills; `down` removes only clusters the harness
  created, and only after the results are pulled.
- The rental price goes into `provider.json` for $/token; SSH machines use
  `--price-per-node-hour`.

## Stages: E2E, P-Only, D-Only

Stage families come from the profile's PD families: one per distinct prefill
(P-Only) or decode (D-Only) role config, named `p-only-<parallelism>x<n>` or
`d-only-...`, filling every GPU of the cluster with that role. The summary's
`source` names the PD family it came from. A profile may also list its own
families with `stage: prefill` (P-Only) or `stage: decode` (D-Only); a derived
family with the same role config is then dropped.

P-Only:

- Each worker runs the prefill role config as an aggregated worker: the
  `disaggregation-*` keys are dropped and decode CUDA graphs are off
  (`--cuda-graph-backend-decode disabled`), as in prefill mode.
- AIPerf caps every request at one output token (`--extra-inputs
  ignore_eos:true max_completion_tokens:1`), so a request is one prefill plus
  one sampled token.
- The objective is input tok/s/GPU (AIPerf `input_token_throughput`).
  Prefix-cache hits count as input; each point's service state records the
  hit rate. A point whose mean OSL exceeds 1.5 tokens is invalid: the cap was
  not honoured.
- No interactivity floors (there is no per-user decode rate), no MTP, no
  accuracy gate.

D-Only:

- Each worker runs the decode role with `--disaggregation-mode decode
  --disaggregation-transfer-backend fake`. `agents/sglang_decode_only.py`,
  started instead of `python3 -m dynamo.sglang` when `CB_DECODE_ONLY=1`,
  gives each request SGLang's fake bootstrap address and registers the worker
  without a prefill peer, so the frontend routes straight to decode. The
  engine allocates the whole prompt's KV pages, skips the transfer and
  decodes: batch size, KV footprint and attention cost are real; the KV
  contents are not. Preflight runs the shim with `--check` inside the image.
- The objective is output tok/s/GPU; the interactivity floors apply.
- MTP runs with simulated acceptance: real drafts over meaningless KV would
  not be accepted.
- The concurrency grid and lane cap widen by the source PD family's total GPUs
  / decode GPUs, so each GPU sees the decode load it carries in PD.
- No accuracy gate.

PD split: once the KB has a P-Only and a D-Only leader for the same cluster,
model, checkpoint, dataset and served context, `INDEX.md` shows
GPUs_P : GPUs_D = (ISL x D) : (OSL x P) and the PD output tok/s/GPU that split
delivers, OSL / (ISL/P + OSL/D), where P is P-Only input tok/s/GPU, D is
D-Only output tok/s/GPU, and ISL/OSL are the D-Only leader's means. It is an
upper bound: KV transfer and latency SLOs are not in it, so confirm the split
with an E2E run.

Not yet verified on GPUs (each report lists the ones that apply):

- the frontend routing to decode-only workers when no prefill worker is
  registered (checked against the Dynamo 1.6.2 and SGLang 0.5.21 / 0.5.23
  sources);
- MoE expert load on decode over meaningless KV, which may differ from real
  requests (EP balance);
- whether `max_completion_tokens:1` holds for every turn of the AgentX
  closed-loop replay (the audit rejects points where it did not);
- whether an aggregated worker with the prefill role config costs what a PD
  prefill worker costs (it sends no KV and has no bootstrap queue).

## Versions

| Component | Version |
|---|---|
| Serving image (default `--image`) | `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.6.2`, pinned by its multi-arch digest: Dynamo 1.6.2, SGLang 0.5.21, CUDA 13.0 (driver >= 580) |
| AIPerf (node0 venv) | 0.14.0 |
| SGLang 0.5.23 | Dynamo main and nightly only: `--image nvcr.io/nvidia/ai-dynamo/sglang-runtime-nightly:<YYYYMMDD-sha>` (pin the dated tag); the harness is checked against the 0.5.23 source too |

SGLang 0.5.21 changed some server arguments; the harness handles them:

- `cuda-graph-bs` and `cuda-graph-max-bs` became `cuda-graph-bs-decode` /
  `-prefill` and `cuda-graph-max-bs-decode` / `-prefill`. Profiles, ladder
  rungs and recipe ConfigMaps may still use the old keys: they are read as the
  `-decode` keys (as Dynamo's profiler maps them); an explicit new key wins.
- `enable-dp-attention` goes on the command line, not in the `--config`
  YAML: SGLang 0.5.23 deprecates the flag and its YAML loader rejects the key,
  and 0.5.21 has no `--attn-dp-size` yet. `rendered_configs` keep the key.
- The startup-log patterns (cold-start phases, KV capacity) and the server
  arguments dump match 0.5.21 and 0.5.23 as well as older releases.

## Datasets

| `--dataset` | Example | AIPerf |
|---|---|---|
| AgentX trace (default) | `semianalysis_cc_traces_weka_062126` (the full corpus; `_256k` is the variant without requests over 256k tokens) | `--scenario agentx --public-dataset NAME`, AIPerf's short command: the preset supplies streaming chat, seed 42, trajectory start 0.25-0.75 and 10 warmup requests per lane, and the scenario enforces the first-turn cache bust and keeps the trace delays (idle gaps capped at 300 s); the harness adds the duration, `--max-context-length` (drops whole traces longer than the served context minus 2048 tokens), server metrics, `--extra-inputs` and the router's session affinity |
| any other AIPerf public dataset | `sharegpt`, `aimo`, `spec_bench` | `--public-dataset NAME`, closed-loop concurrency |
| a file on node0 | `mooncake_trace:/root/cluster_bench/inputs/t.jsonl` (`file:PATH` means `mooncake_trace`; a `weka_trace` file is replayed by the AgentX scenario) | `--input-file PATH --custom-dataset-type TYPE`; closed-loop concurrency, trace timestamps are not replayed (`--no-fixed-schedule`) |
| synthetic prompts | `synthetic:isl=8192,osl=1024[,isl_std=0,osl_std=0]` | `--synthetic-input-tokens-mean/-stddev`, `--output-tokens-mean/-stddev` |

AgentX runs and AIPerf's verdict (`metadata.submission_valid` in
`profile_export_aiperf.json`):

- screen points (search, pilot, NIC sweep) measure 480 s after a 180 s cache
  warmup (`--agentic-cache-warmup-duration`). That is below the 900 s AgentX
  floor, so they pass `--unsafe-override` and AIPerf marks them
  `submission_valid=false`; they only rank configurations.
- The confirm run keeps the preset's warmup and passes
  `--benchmark-duration 1200` without `--unsafe-override`: AIPerf honours any
  explicit duration >= 900 s, so it is a valid AgentX run.
  `--confirm-duration-s 3600` uses the preset's own length (about 40 more
  minutes of budget).
- The report prints AIPerf's verdict on the confirm run and its reasons. A run
  AIPerf marks incomplete (`is_complete=false`) is invalid.
- A local `weka_trace` file always needs `--unsafe-override` (the corpus
  cannot be pinned).
- P-Only and D-Only change the AgentX load (one output token per request; no
  real prefill or KV transfer), so their runs are never AgentX submissions,
  even when AIPerf stamps them valid.

`controller sync --upload FILE` (also on `run-all`, repeatable) copies a local
file to `<workdir>/inputs/<name>` on node0 and checks its sha256.
`--aiperf-arg=ARG` (repeatable) appends `ARG` to every `aiperf profile`;
flags the harness manages (load, dataset, output length, model, URL,
artifacts) are refused. Preflight checks the dataset's flags against
`aiperf profile --help`.

## How a run works

| Phase | What happens |
|---|---|
| preflight | Replicate code to peers, bootstrap venvs, probe GPUs/HCAs/topology/disks, pick profiles (for P-Only/D-Only the stage families) and a local weights disk (never `/mnt` on Lium), pull the image, verify every SGLang/Dynamo/AIPerf flag (the stage's and the dataset's included) against `--help` of the pinned versions; D-Only: run the decode-only shim's `--check` inside the image |
| download | Resolve the HF commit, download 1/N of the shards per node in the background, meanwhile run intra-node P2P microbenchmarks and a dummy-weight kernel smoke (walks the GPU profile's ladder), then NCCL-broadcast the missing shards over the RDMA fabric and verify sha256; then measure the weights disk's direct-read speed |
| fabric | Upstream switch topology and link health (see below): identify (sysfs / MAD / LLDP), traces between every node pair per rail, a perftest link matrix (round-robin pairs, every rail concurrently, ~2-4 min on 8 nodes; `--no-link-matrix` skips it), and the list of switch ports sampled at every load point |
| microbench | Inter-node NCCL collectives and GPU-direct send/recv with 8/4/2/1 NICs (speed-of-light references); the intra-node collectives run with `NCCL_DEBUG=INFO` (`INIT`), and the transport NCCL set up for every GPU-to-GPU hop (`P2P/...`, or `SHM/...` through host memory) is kept with the peak and shown next to `nvidia-smi topo -m` in report section 2.1 |
| search | Topology families in priority order (recipe baseline first), powers-of-two concurrency climb per family, pruning, P:D alternate by symptom, floor bracketing, MTP-off ablation (P-Only: no floors and no MTP; D-Only: MTP with simulated acceptance) |
| accuracy | E2E only (P-Only and D-Only produce no real answers). GSM8K 5-shot (200 questions, sha256-pinned test split, greedy, Dynamo `/v1/completions`) on the leader; a configuration below the floor (0.80) or more than 0.07 below the best checked one (outside the 95% CI) never leads, so the next best is checked (`--accuracy-max-checks`, default 3); `--no-accuracy` disables it |
| pilot | n=3 repetitions at the leader's knee: noise floor and minimum detectable effect |
| nic_sweep | 1/2/4/8 NICs on the best configuration that has measured cross-node traffic |
| confirm | Rerun of the leader with the AgentX preset's warmup, `--confirm-duration-s` (default 1200 s, >= 900 s for AgentX) and no `--unsafe-override` |
| nvlink | NVLink against PCIe: tear down, switch NVLink out of the GPU-GPU path on every node (Hopper: `nvidia-smi nvlink -sBwMode OFF`, "P2P is via PCIe bus"; Blackwell: `-sLWidth 0` only if offered), prove it with a GPU-GPU copy benchmark (must drop to PCIe speed), deploy the leader with NVLS and SGLang's custom all-reduce off (`NCCL_P2P_LEVEL=SYS`), measure one point at the leader's concurrency, restore NVLink (also on resume after a crash and in finalize). Without root or a supported switch: NCCL without P2P, staged through host memory (a labelled lower bound). A DeepEP leader (NVLink-only intranode kernels) hands the comparison to the best family without DeepEP; PCIe-only GPUs skip it. The run is invalid if NVLink counters show traffic. Its budget is the first to yield: the leader's climb and every earlier step may spend it (`--no-nvlink-ablation` turns it off) |
| diagnose | Opt-in (`--diagnose`): the leader redeployed as a diagnostic variant (SGLang `--enable-mfu-metrics`, device timer, `--expert-balancedness-report-mode prometheus` for MoE; workers under Nsight Systems from the pinned image or a derived `cb-diag/...` image built on each node), a short load, `POST /engine/control/start_profile` for 10 forward steps, `nsys stats` -> kernel-time shares per role, and each report's SQLite export reduced on the node to the wall-time split of a forward step (NCCL, attention, MoE, dense GEMM, other, idle; stalls >= 5 ms cut out for a steady split). The report draws it: step bars per role, every GPU's idle over the window with the long collectives on it, two steps of one GPU's timeline as stacked shares, and a 1.8 ms close-up with every kernel at its real time. Never ranked |
| finalize | "after" health snapshot, summary, report, KB export |

## Focus resources of the leader

Report section 2 and the KB view `resources.md` answer these questions about
the leader at its saturation point (2.1 interconnect, 2.2 NVLink against
PCIe, 2.3 MFU and MBU, 2.4 KV capacity), each with the method stated next to the
number:

| Question | How it is derived | Where |
|---|---|---|
| How much interconnect does it need? | For each link kind (NVLink per GPU, GPU PCIe, GPU<->NIC PCIe, IB/RoCE per node) the busiest device's p99 over 1-s bins, worse direction; utilization against the nominal per-direction rate; need = p99 / `--link-target-util` (default 0.7). Fewest NICs per node twice: measured by the NIC sweep (fewest within the noise floor with every larger count also holding; the sweep runs on the best cross-node family when the leader has none, and is attributed to it) and estimated from the leader's own IB need / one NIC's rate. NVLink is needed when the GPU-GPU need exceeds one PCIe link | `summary.interconnect`, KB `index.jsonl` (`min_nics_*`, `ib_*`, `nvlink_*`) |
| Does it need NVLink? | Measured: the same configuration at the same concurrency with NVLink switched off (phase `nvlink` above): tok/s/GPU, per-user rate, TTFT and ITL against NVLink, the delta judged against the noise floor, the method, what carried the traffic (PCIe P2P or host staging), the GPU-GPU copy rate and PCIe p99 during the run | `summary.nvlink`, KB `index.jsonl` `nvlink_off_*`, `run.jsonl` `p2p` |
| How much of the GPUs' floating-point peak does it use (MFU)? | Model FLOPs from the checkpoint's `config.json` (MLA, MoE with shared experts and dense layers, DSA top-k attention plus its indexer, dense GQA) for useful tokens only: prompt tokens not served from the prefix cache (per-request `usage` or the server's hit rate) and generated tokens; MTP drafts are excluded. Divided by AIPerf's throughput duration (its profiling phase, grace period included) x GPUs x the GPU profile's dense peak at the checkpoint's GEMM precision; per role for PD; NVML GPM tensor activity alongside | `PointResult.mfu`, `summary.model_arch`, KB `run.jsonl` `mfu_*`, `index.jsonl` `leader_mfu_*` |
| How much of the GPUs' memory bandwidth does it use (MBU)? | Measured, not modeled: NVML GPM `DRAM_BW_UTIL` (bandwidth used / the GPU's maximum) of every GPU at 1 s from the telemetry agent, the mean over the point per role, GPU-weighted across PD roles; achieved TB/s = MBU x the GPU profile's `hbm_tbs`. No bytes model: the experts a batch routes to, KV replicated across TP ranks, the DSA indexer scan and speculative decoding change the bytes a step reads. Needs GPM (Hopper and newer); without samples the report says MBU was not measured. MBU does not need `config.json`, so a point without model FLOPs still has it | `PointResult.mfu` (`roles[].mbu`, `total_mbu`, `*_mem_tbs_per_gpu`), KB `run.jsonl` `mbu_*`, `index.jsonl` `leader_mbu_*`, `peak_mem_tbs_per_gpu` |
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
| Service state | Dynamo frontend / router and SGLang worker `/metrics` scraped by AIPerf (names checked against the SGLang 0.5.21 and 0.5.23 sources): running/queued requests, KV usage, prefix hit rate, PD queues, KV transfer (latency, bootstrap, allocation wait), per-request stage latencies (`per_stage_req_latency_seconds`: bootstrap, prefill, KV transfer, decode wait), CUDA graph coverage of decode passes, retracted and evicted tokens, request-weighted MTP acceptance (generated tokens per verify step) and acceptance rate; bottleneck label with evidence; energy per output token | every point | `points/<id>/servicestate.json`, `PointResult.state` |
| GPU activity | NVML GPM per GPU at 1 s: SM, occupancy, tensor, DRAM bandwidth, PCIe and NVLink rates; regime per role (compute / memory / waiting / idle) for the bottleneck label; the DRAM bandwidth is the MBU of report section 2.3; the report splits each role's time three ways (SMs at work = SM activity, a kernel running on idle SMs = graphics-engine minus SM activity, no kernel = the rest) for every point of the leader's family, and turns PCIe TX into MB per output token (input token for prefill); per-GPU availability (`unsupported`, `NVML_ERROR_*`) is data | every point (telemetry agent; off while nsys runs) | `stress.json` `activity`, `PointState.stress.activity` |
| Requests | From AIPerf's per-request records (`--export-level records`, no extra load): goodput under SLO tiers (`u50`: TTFT <= 10 s and ITL <= 20 ms; `u20`: 30 s / 50 ms; `--goodput-slo`) over the same duration AIPerf's throughput divides by, so it never exceeds the throughput, the best point per tier, TTFT by session turn (prefix reuse), session spans and subagent requests | every point | `points/<id>/requests.json`, `PointResult.requests` |
| Accuracy | GSM8K result, Wilson 95% CI, verdict and reason per checked configuration; answer source (text / reasoning field) | accuracy phase | `accuracy/`, `summary.accuracy`, KB `accuracy.jsonl` |
| Diagnostics (opt-in) | kernel-time shares by category (attention, MoE GEMM, dense GEMM, DeepEP, NCCL, KV cache, ...) and top kernels per role (with a reading of the name from `kernel_labels` in the locale file, average µs and calls per step per GPU), the wall-time split of a step per role, the longest idle gaps and calls, per-GPU idle bins and a two-step sample of the timeline, per-GPU trace coverage, nsys version; SGLang estimated TFLOP/s and memory traffic per GPU, forward-busy share, EPLB balancedness | diagnose phase | `diagnostics/`, `summary.diagnostics`, KB `index.jsonl` `diag_*` |
| Cold start | per container: time to weights loaded, KV pool, CUDA graphs, ready; weight/KV memory, `max_total_num_tokens`, `max_running_requests`; NCCL/OOM/RDMA log issues | every deployment | `deployments/<fp>/lifecycle.json`, `DeployResult.lifecycle` |
| Node facts and health | PCIe links (incl. upstream bridges, AER/ACS), NUMA, hugepages/THP, `/dev/shm`, memlock, cgroup, IOMMU, kernel modules, driver flavour, `nvidia-smi -q -x` (ECC, row remap, power limits, clocks, fabric/clique), NVLink status/errors, P2P matrices, HCA counters, kernel log (Xid, mlx5) when readable; "before" and "after" snapshots -> what grew under load | preflight, finalize | `facts/`, `summary.fact_findings`, `health_deltas`, report section 11, KB `health.jsonl` |
| Environment | software stack inside the pinned image (sglang, flashinfer, torch/NCCL, nixl, dynamo, UCX, OS), clock offsets between nodes, weights-disk read speed | preflight, download | `summary.software`, `clock_offsets`, `disk_reads` |
| Cost | lium fabric id/type/link rate and price (controller `provider.json`, no secrets or IPs) or `--price-per-node-hour` (SSH machines): cluster $/h, experiment cost, $/1M output tokens at saturation and at the interactivity floors | finalize | `summary.provider`, `summary.cost`, KB `index.jsonl` |

Node bootstrap installs `infiniband-diags ibverbs-utils perftest` with apt when
they are missing on a pod (best effort, once; `CB_FABRIC_TOOLS=0` disables it),
so MAD queries and the link matrix work even if the image lacks them.

Key facts the harness relies on (verified in the AIPerf 0.14.0 wheel): every
AIPerf run cache-busts first-turn prefixes with a fresh uuid, so concurrency
points need no worker restart; `--max-context-length` drops whole traces, so
Hopper profiles serve 262144 tokens instead of the recipe's 250000. The RTX PRO
6000D profile serves 229376: a TP8 decode worker there holds 248192 tokens of
KV, and AIPerf aborts an AgentX run when one warmup request is longer than that.
When AIPerf leaves no summary export, the point's audit names AIPerf's abort
reason and the most common request error.

## Layout

```text
domain/    pure models and logic, no I/O (schema, profiles, stages (P-Only/D-Only families, worker
           config, PD split), datasets (--dataset forms), inventory, nicplan, search, analysis, bandwidth,
           fabric, nodefacts, servicestate, loadstress, lifecycle, pointstate, switchload, cost, requests,
           gpuactivity, accuracy, kernels, modelarch (config.json FLOPs/KV bytes, MFU, MBU), capacity
           (interconnect needs, fewest NICs, KV capacity), nvlink (switch methods, NVLink-off comparison))
ports.py   protocols the application depends on (Executor, Cluster, Provider, Engine, TransportPlugin, Clock)
fsutil.py  crash-safe file primitives shared by every layer (atomic writes, file lock, sha256)
adapters/  profile_store (YAML profiles + recipe ConfigMaps), ssh/docker executors, lium and hosts-file
           providers, machines (SSH machines: access lines, discovery, control addresses, trust key),
           SGLang renderer, transports, AIPerf, HF fetch, telemetry, microbenchmarks,
           fabric prober, node-facts collector, accuracy probe runner, Nsight Systems diagnostics
agents/    standalone node-side scripts (probe, telemetry sampler with NVML GPM, P2P/NCCL benchmarks,
           NCCL file gather, fabric_probe, node_facts, accuracy_probe, sglang_decode_only (D-Only worker),
           sglang_hotfix (named SGLang source edits a profile applies at worker launch))
app/       event store, deployment session, pipeline and phases (incl. accuracy gate, diagnose), fabric/facts
           collection points, model_ops (config.json read once per run, MFU per point), nvlink_ops
           (switch NVLink off on every node and back), summary, simulator
controller.py  laptop side: SSH machines or a lium rental; sync, launch, watch, pull, down
report/    Chinese Markdown/HTML report (stage wording: "stages" in i18n/zh_CN.yaml)
kb/        knowledge-base records and writer
profiles/  gpus/*.yaml, models/*.yaml, families/<model>/<gpu>.yaml (recipe ConfigMaps + overrides + ladder)
```

Adding a GPU type or model is YAML only. A new transport is one module in
`adapters/transports/` registered in `REGISTRY`, plus its name in the
`Transport` literal of `domain/schema.py`. A new cluster source implements
`ports.Provider`. A platform special case like Lium's is a `--platform`
value: `PLATFORM_FORBIDDEN_PREFIXES` in `domain/inventory.py` and the
provider choice in `cli.py`.

`summary.json` and the knowledge base keep English text; the Chinese report
localizes known reason templates at render time (`report/reasons.py`).

## Tests

```bash
# in a dev container
python -m pytest --confcutdir=tests/benchmarks tests/benchmarks/cluster_bench
# anywhere with uv (no GPU, network or cluster needed)
uv run --no-project --python 3.12 --with pytest==8.4.2 --with pytest-benchmark \
    --with pytest-asyncio --with pydantic --with pyyaml --with huggingface_hub \
    python -m pytest --confcutdir=tests/benchmarks tests/benchmarks/cluster_bench -q
```

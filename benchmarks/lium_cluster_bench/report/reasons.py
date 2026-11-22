# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Localize machine-generated reason strings for the Chinese report.

The planner, the audit and the bandwidth analysis write short English
reasons into the summary (the knowledge base keeps them verbatim). The
report translates the known templates; anything unrecognised passes through
unchanged rather than being dropped.
"""

from __future__ import annotations

import re
from typing import Callable

SYMPTOMS = {
    "prefill_bound": "瓶颈在 prefill",
    "decode_bound": "瓶颈在 decode",
    "balanced": "prefill 和 decode 比较均衡",
    "unknown": "瓶颈不明",
}
FABRIC_KINDS = {
    "infiniband": "InfiniBand",
    "roce": "RoCE",
    "mixed": "IB + RoCE 混合",
    "unknown": "未知类型",
}
FAILURES = {
    "oom": "显存不足",
    "kernel_unsupported": "kernel 不支持",
    "rdma_registration": "RDMA 内存注册失败",
    "nccl": "NCCL 错误",
    "port_in_use": "端口被占用",
    "timeout": "启动超时",
    "container_exit": "容器退出",
    "smoke_failed": "冒烟请求失败",
    "engagement": "配置未生效",
    "unknown": "未知错误",
}

Rule = tuple[re.Pattern[str], Callable[[re.Match[str]], str]]


def _rule(pattern: str, render: Callable[[re.Match[str]], str]) -> Rule:
    return re.compile(pattern), render


ACCURACY_STATUS = {"pass": "通过", "fail": "未通过", "error": "无法判定", "skipped": "已跳过"}
# "gsm8k 5-shot accuracy 0.940 (95% CI 0.898-0.965, n=200)" (domain/accuracy.py)
_SCORE = r"(\w+) (\d+)-shot accuracy ([\d.]+) \(95% CI ([\d.]+)-([\d.]+), n=(\d+)\)"


def _score(m: re.Match[str], first: int = 1) -> str:
    bench, shots, acc, lo, hi, n = (m[first + i] for i in range(6))
    return f"{bench.upper()} {shots}-shot 准确率 {acc}（95% CI {lo}–{hi}，n={n}）"


def _floor_text(text: str) -> str:
    found = re.match(r"passes floor ([\d.]+)", text)
    if found:
        return f"达到下限 {found[1]}"
    return "未设下限" if text == "has no floor" else text


RULES: tuple[Rule, ...] = (
    # fabric topology (domain/fabric.topology_findings, analyze_link_matrix)
    _rule(
        r"^(infiniband|roce|mixed|unknown) fabric: (\d+) rail\(s\) x (\S+) Gb/s per node \((\d+) nodes\)$",
        lambda m: f"{FABRIC_KINDS.get(m[1], m[1])} 网络：每节点 {m[2]} 条 rail × {m[3]} Gb/s（{m[4]} 个节点）",
    ),
    _rule(
        r"^rail-optimized: (\d+) rails, (\S+) nodes per leaf$",
        lambda m: f"rail-optimized：{m[1]} 条 rail，每台 leaf 下 {m[2]} 个节点",
    ),
    _rule(
        r"^not rail-optimized: a leaf carries up to (\d+) rails of our HCAs \((\d+) leaves, (\S+) nodes per leaf\)$",
        lambda m: f"不是 rail-optimized：一台 leaf 最多接了本集群 {m[1]} 条 rail 的网卡（{m[2]} 台 leaf，每台下 {m[3]} 个节点）",
    ),
    _rule(
        r"^leaf switches unknown \(no MAD access and no LLDP\)$",
        lambda m: "不知道 leaf 交换机是哪台（没有 MAD 权限，也没收到 LLDP）",
    ),
    _rule(
        r"^rails are isolated subnets(?: \((\d+) SMs\))?$",
        lambda m: "每条 rail 是独立子网" + (f"（{m[1]} 个 SM）" if m[1] else ""),
    ),
    _rule(r"^rails share one subnet$", lambda m: "所有 rail 在同一个子网"),
    _rule(
        r"^(\d+)-tier path between our nodes: up to (\d+) switch hop\(s\) \((.*)\)$",
        lambda m: f"节点之间经过 {m[1]} 层交换机：最多 {m[2]} 跳（{m[3]}）",
    ),
    _rule(
        r"^up to (\d+) switch hop\(s\) between the node pairs with known paths \((.*)\); other pairs unknown$",
        lambda m: f"已知路径的节点对之间最多 {m[1]} 跳（{m[2]}）；其他节点对不清楚",
    ),
    _rule(
        r"^non-blocking leaves: host-facing capacity <= uplink capacity on (\d+) censused leaves$",
        lambda m: f"leaf 无收敛：已统计的 {m[1]} 台 leaf 下行容量都不超过上行容量",
    ),
    _rule(
        r"^(\d+) of (\d+) censused leaves oversubscribed, worst (.+) ([\d.]+):1 \((\d+) host ports vs (\d+) uplinks\)$",
        lambda m: f"{m[1]}/{m[2]} 台已统计的 leaf 有收敛，最严重的 {m[3]} 收敛比为 {m[4]}:1（{m[5]} 个主机端口对 {m[6]} 个上行端口）",
    ),
    _rule(
        r"^(\d+) of (\d+) leaves also serve other tenants' hosts \(up to (\d+) host ports per leaf\)$",
        lambda m: f"{m[1]}/{m[2]} 台 leaf 还接了其他租户的主机（每台最多 {m[3]} 个主机端口），上行带宽可能被共用",
    ),
    _rule(r"^degraded link: (.*)$", lambda m: f"链路降级：{m[1]}"),
    _rule(
        r"^switch DCBX advertises PFC on priority (.*)$",
        lambda m: f"交换机 DCBX 通告的 PFC 优先级：{m[1]}",
    ),
    _rule(
        r"^rail (\d+) latency groups: (.*)$",
        lambda m: f"rail {m[1]} 的延迟分组："
        + re.sub(
            r"(\d+) pair\(s\) ~([\d.]+) us \(hops (\S+)\)",
            lambda g: f"{g[1]} 对约 {g[2]} μs（{g[3]} 跳）",
            m[2],
        ),
    ),
    # planner (domain/search.py)
    _rule(r"^climb complete$", lambda m: "加并发结束"),
    _rule(r"^budget stopped the climb before saturation$", lambda m: "预算用完，还没加到饱和"),
    _rule(
        r"^grid max (\d+) reached while still rising$",
        lambda m: f"到并发上限 {m[1]} 时吞吐仍在上升",
    ),
    _rule(
        r"^alternate not selected \(primary (\S+): (\w+)\)$",
        lambda m: f"没选这个备选：主拓扑 {m[1]} {SYMPTOMS.get(m[2], m[2])}",
    ),
    _rule(
        r"^best ([\d.]+) < ([\d.]+) x leader ([\d.]+) tok/s/GPU$",
        lambda m: f"最高 {m[1]} tok/s/GPU，不到当前最优 {m[3]} 的 {m[2]} 倍，提前放弃",
    ),
    _rule(r"^no valid measurement \((.*)\)$", lambda m: f"没有有效的测量结果（{localize(m[1])}）"),
    _rule(r"^insufficient budget for deploy \+ first point$", lambda m: "预算不够部署并测第一个点"),
    _rule(r"^image lacks flags (.*)$", lambda m: f"镜像缺少参数 {m[1]}"),
    _rule(
        r"^opt-in family \(enable with --families\)$", lambda m: "默认不测，需用 --families 指定"
    ),
    _rule(r"^not reached$", lambda m: "因预算没有执行"),
    _rule(
        r"^(oom|kernel_unsupported|rdma_registration|nccl|port_in_use|timeout|"
        r"container_exit|smoke_failed|engagement|unknown): ?(.*)$",
        lambda m: f"{FAILURES[m[1]]}：{m[2]}" if m[2] else FAILURES[m[1]],
    ),
    # saturation (domain/analysis.py)
    _rule(r"^fewer than two valid concurrency points$", lambda m: "有效的并发点不到两个"),
    _rule(
        r"^peak at c=(\d+); c=(\d+) is ([+-][\d.]+)%$",
        lambda m: f"峰值在 C={m[1]}；C={m[2]} 处为 {m[3]}%",
    ),
    _rule(
        r"^top two points within ([\d.]+)% \(([+-][\d.]+)%\)$",
        lambda m: f"顶部两点相差 {m[2]}%，在 {m[1]}% 的测量噪声以内",
    ),
    _rule(
        r"^still rising ([+-][\d.]+)% at c=(\d+)$", lambda m: f"C={m[2]} 时仍在上升（{m[1]}%）"
    ),
    # NIC sweep base (domain/search.py)
    _rule(
        r"^the throughput leader has cross-node data-plane traffic$",
        lambda m: "吞吐最优的配置有跨节点流量",
    ),
    _rule(
        r"^no measured configuration has cross-node data-plane traffic; NIC count cannot matter$",
        lambda m: "测过的配置都没有跨节点流量，网卡数量影响不到吞吐",
    ),
    _rule(
        r"^the throughput leader has no cross-node data-plane traffic, so NIC count is "
        r"irrelevant for it; the sweep uses the best cross-node configuration \((\S+)\)$",
        lambda m: f"吞吐最优的配置没有跨节点流量，网卡数量影响不到它；网卡数量对比改在有跨节点流量的配置里最好的 {m[1]} 上做",
    ),
    # audit (domain/analysis.py)
    _rule(r"^no AIPerf export found$", lambda m: "找不到 AIPerf 的导出结果"),
    _rule(
        r"^aiperf (\S+) != pinned (\S+)$",
        lambda m: f"AIPerf 版本 {m[1]} 和固定的版本 {m[2]} 不一致",
    ),
    _rule(r"^run was cancelled$", lambda m: "运行被取消"),
    _rule(r"^no output_token_throughput$", lambda m: "缺少 output_token_throughput"),
    _rule(
        r"^(\d+) completed requests < (\d+) lanes$",
        lambda m: f"完成的请求数 {m[1]} 少于并发数 {m[2]}",
    ),
    _rule(r"^error rate (\S+) >= (\S+)$", lambda m: f"错误率 {m[1]} ≥ {m[2]}"),
    _rule(
        r"^context overflow (\d+)/(\d+) > 1% \(AgentX invalid\)$",
        lambda m: f"超出上下文长度的会话 {m[1]}/{m[2]}，超过 1%（AgentX 判为无效）",
    ),
    _rule(
        r"^measured (\d+)s < (\d+)% of (\d+)s$",
        lambda m: f"实际测了 {m[1]} s，不到 {m[3]} s 的 {m[2]}%",
    ),
    _rule(r"^config not engaged: (.*)$", lambda m: f"配置未生效：{m[1]}"),
    _rule(
        r"^a serving container exited or restarted during the run$",
        lambda m: "运行期间有服务容器退出或重启",
    ),
    _rule(
        r"^NIC restriction not engaged \(traffic on non-selected HCAs\)$",
        lambda m: "网卡限制没生效（没选用的网卡上也有流量）",
    ),
    _rule(
        r"^NIC engagement not provable \(no cross-node traffic\)$",
        lambda m: "没有跨节点流量，无法确认网卡限制已生效",
    ),
    _rule(r"^telemetry coverage (\S+)$", lambda m: f"监控数据覆盖率 {m[1]}"),
    # summary conditions, limitations and unverified items (app/summary.py)
    _rule(
        r"^AIPerf client co-located on node0 \(no spare CPU host\)$",
        lambda m: "AIPerf 压测客户端和服务都在 node0 上（没有单独的 CPU 机器）",
    ),
    _rule(
        r"^screening runs use --unsafe-override \(AgentX submission_valid=false\); the confirm run does not$",
        lambda m: "screen 规格的测量带 --unsafe-override（AgentX 标记 submission_valid=false），长时复测不带",
    ),
    _rule(
        r"^served context (\d+) tokens; traces whose recorded peak exceeds (\d+) tokens are dropped by AIPerf$",
        lambda m: f"服务支持的上下文长度 {m[1]} tokens；trace 里峰值超过 {m[2]} tokens 的会话会被 AIPerf 整条丢弃",
    ),
    _rule(
        r"^NIC counts restricted in software \(per-transport env/config\), verified with per-HCA counters$",
        lambda m: "网卡数量通过各传输组件的环境变量或配置来限制，并用每张网卡的计数器确认",
    ),
    _rule(
        r"^simulated speculative acceptance \(SGLANG_SIMULATE_ACC_\*\) enabled; results do not reflect measured acceptance$",
        lambda m: "开启了模拟的投机解码接受率（SGLANG_SIMULATE_ACC_*），结果不代表实测接受率",
    ),
    _rule(
        r"^DRY RUN: all measurements are synthetic and not hardware evidence$",
        lambda m: "DRY RUN：所有测量都是模拟数据，不能当作硬件测试结果",
    ),
    _rule(
        r"^noise floor not measured; a provisional 5% threshold was used$",
        lambda m: "没测到测量噪声，暂用 5% 作为判断阈值",
    ),
    _rule(
        r"^throughput curve not saturated: (.*)$", lambda m: f"吞吐曲线没饱和：{localize(m[1])}"
    ),
    _rule(
        r"^confirmatory rerun invalid: (.*)$",
        lambda m: f"长时复测无效：{'；'.join(localize(r) for r in m[1].split('; '))}",
    ),
    _rule(
        r"^MTP disabled by fallback rung (\S+); MTP on/off not compared$",
        lambda m: f"fallback 规则 {m[1]} 已关闭 MTP，所以没做 MTP 开 / 关对比",
    ),
    _rule(
        r"^MTP already disabled by fallback rung (\S+)$",
        lambda m: f"fallback 规则 {m[1]} 已关闭 MTP",
    ),
    _rule(
        r"^SGLang 0\.5\.18 Mooncake device flag --disaggregation-ib-device \(checked by the preflight --help probe\)$",
        lambda m: "SGLang 0.5.18 的 Mooncake 网卡参数 --disaggregation-ib-device（环境预检时用 --help 核对）",
    ),
    _rule(
        r"^NVML NVLink throughput counters on this driver/GPU \(falls back to nvidia-smi or 'unavailable'\)$",
        lambda m: "当前驱动 / GPU 上的 NVML NVLink 流量计数器（不可用时改用 nvidia-smi，或标为不可用）",
    ),
    _rule(
        r"^SGLang speculative acceptance-length metric name \(matched by substring\)$",
        lambda m: "SGLang 投机解码接受长度的指标名（按子串匹配）",
    ),
    _rule(
        r"^IB port counters assumed in 4-byte units, calibrated against NCCL microbenchmarks when available$",
        lambda m: "IB 端口计数器按 4 字节为单位换算，能测时用 NCCL 带宽峰值测试校准",
    ),
    _rule(
        r"^PortXmitWait tick length \(switch and HCA stall fractions are estimates\)$",
        lambda m: "PortXmitWait 计数的时间单位（交换机和网卡的发送等待占比是估算值）",
    ),
    _rule(
        r"^SGLang 0\.5\.18 / Dynamo 1\.5\.1 metric names for KV usage and PD queues \(absent metrics are reported as missing\)$",
        lambda m: "SGLang 0.5.18 / Dynamo 1.5.1 里 KV 占用和 PD 队列的指标名（采不到的指标会标为缺失）",
    ),
    _rule(
        r"^SGLang startup log lines used for cold-start phases and KV capacity$",
        lambda m: "用来解析冷启动各阶段和 KV 容量的 SGLang 启动日志格式",
    ),
    _rule(
        r"^Fabric tools in the pinned image \(infiniband-diags, perftest\); probing falls back to the host tools, then to sysfs only$",
        lambda m: "固定版本的镜像里有没有网络诊断工具（infiniband-diags、perftest）；没有时改用主机上的工具，再不行只用 sysfs",
    ),
    _rule(r"^fabric probing failed: (.*)$", lambda m: f"网络拓扑探测失败：{m[1]}"),
    _rule(
        r"^families not applicable to (\d+) nodes x (\d+) GPUs: (.*)$",
        lambda m: f"以下拓扑不适用于 {m[1]} 节点 × {m[2]} GPU，已跳过：{m[3]}",
    ),
    _rule(r"^node facts \((\w+)\) failed: (.*)$", lambda m: f"节点信息采集（{m[1]}）失败：{m[2]}"),
    _rule(
        r"^software stack / clock offsets failed: (.*)$",
        lambda m: f"软件版本 / 时钟偏移采集失败：{m[1]}",
    ),
    _rule(r"^weights disk read failed: (.*)$", lambda m: f"权重盘读取速度测试失败：{m[1]}"),
    _rule(
        r"^deploy lifecycle unavailable for (\S+): (.*)$",
        lambda m: f"部署 {m[1]} 的启动过程解析失败：{m[2]}",
    ),
    _rule(r"^fabric topology unreadable: (.*)$", lambda m: f"网络拓扑数据无法读取：{m[1]}"),
    _rule(r"^node facts unreadable: (.*)$", lambda m: f"节点信息无法读取：{m[1]}"),
    _rule(
        r"^planner stalled: (\S+) c=(\d+) issued (\d+) times$",
        lambda m: f"规划器卡住了：{m[1]} 在 C={m[2]} 已重复下发 {m[3]} 次",
    ),
    # accuracy gate verdicts (domain/accuracy.accuracy_verdict) and summary notes
    _rule(
        rf"^{_SCORE} below floor ([\d.]+) even if all (\d+) failed requests were correct$",
        lambda m: f"{_score(m)}，即使 {m[8]} 个失败请求全部答对也低于下限 {m[7]}",
    ),
    _rule(
        rf"^{_SCORE} below floor ([\d.]+)$",
        lambda m: f"{_score(m)}，低于下限 {m[7]}",
    ),
    _rule(
        rf"^{_SCORE} is ([\d.]+) below the best checked ([\d.]+) \(max drop ([\d.]+)\)$",
        lambda m: f"{_score(m)}，比已通过的最好成绩 {m[8]} 低 {m[7]}（最多允许低 {m[9]}）",
    ),
    _rule(
        rf"^{_SCORE} (passes floor [\d.]+|has no floor); ([\d.]+) below the best checked ([\d.]+) but within the 95% CI$",
        lambda m: f"{_score(m)}，{_floor_text(m[7])}；比已通过的最好成绩 {m[9]} 低 {m[8]}，但在 95% 置信区间内",
    ),
    _rule(
        rf"^{_SCORE} (passes floor [\d.]+|has no floor); within ([\d.]+) of the best checked ([\d.]+)$",
        lambda m: f"{_score(m)}，{_floor_text(m[7])}；和已通过的最好成绩 {m[9]} 相差不超过 {m[8]}",
    ),
    _rule(
        rf"^{_SCORE} (passes floor [\d.]+|has no floor)$",
        lambda m: f"{_score(m)}，{_floor_text(m[7])}",
    ),
    _rule(
        r"^accuracy probe: (\d+) of (\d+) requests failed \(error rate ([\d.]+) > ([\d.]+)\)$",
        lambda m: f"精度检查：{m[2]} 个请求中 {m[1]} 个失败（失败率 {m[3]}，超过 {m[4]}），无法判定",
    ),
    _rule(
        r"^accuracy probe: no answered questions \((\d+) of (\d+) requests failed\)$",
        lambda m: f"精度检查：一道题都没答（{m[2]} 个请求中 {m[1]} 个失败）",
    ),
    _rule(r"^accuracy probe failed: (.*)$", lambda m: f"精度检查程序运行失败：{m[1]}"),
    _rule(r"^accuracy check skipped: (.*)$", lambda m: f"已跳过精度检查：{m[1]}"),
    _rule(
        r"^accuracy gate excluded (\S+): (.*)$",
        lambda m: f"精度检查淘汰了 {m[1]}：{localize(m[2])}",
    ),
    _rule(
        r"^accuracy gate disabled: the leader's numerics are unchecked$",
        lambda m: "精度检查已关闭：最优配置算得对不对没有验证",
    ),
    _rule(r"^the leader was not accuracy-checked$", lambda m: "最优配置没做精度检查"),
    _rule(
        r"^leader accuracy check (\w+): (.*)$",
        lambda m: f"最优配置的精度检查结果是“{ACCURACY_STATUS.get(m[1], m[1])}”：{localize(m[2])}",
    ),
    # diagnostic window (app/phases/diagnose.py)
    _rule(
        r"^Nsight Systems unavailable on (\d+) node\(s\): (.*)$",
        lambda m: f"{m[1]} 个节点上 Nsight Systems 不可用：{m[2]}",
    ),
    _rule(r"^diagnostic deployment failed: (.*)$", lambda m: f"诊断配置部署失败：{m[1]}"),
    _rule(r"^diagnostic load failed: (.*)$", lambda m: f"诊断时段的压测失败：{m[1]}"),
    _rule(
        r"^start_profile on (\S+) failed: (.*)$",
        lambda m: f"向 {m[1]} 发送 start_profile 失败：{m[2]}",
    ),
    _rule(
        r"^(\d+) Nsight Systems report\(s\) not ready$",
        lambda m: f"{m[1]} 份 Nsight Systems 报告没有生成",
    ),
    _rule(
        r"^no kernel breakdown: no readable Nsight Systems report$",
        lambda m: "没有 kernel 耗时拆分：没有可读的 Nsight Systems 报告",
    ),
    _rule(
        r"^budget skipped the diagnostic window$",
        lambda m: "预算不够，跳过了诊断时段",
    ),
    # round 3: GPU regimes, engine-internal ratios, per-request records
    _rule(
        r"^(?:also \S+: )?(\S+) GPUs (memory-bandwidth|compute)-bound: (.*)$",
        lambda m: f"{m[1]} GPU 的瓶颈在{'显存带宽' if m[2] == 'memory-bandwidth' else '计算'}："
        f"{localize_activity(m[3])}",
    ),
    _rule(
        r"^(?:also \S+: )?(\S+) GPUs (waiting|idle) with no queue anywhere: (.*)$",
        lambda m: f"{m[1]} GPU {REGIMES[m[2]]}且各处都没有排队：{localize_activity(m[3])}",
    ),
    _rule(
        r"^(?:also \S+: )?(\S+): only (\S+) of decode passes replayed a CUDA graph \(running batch above the largest captured size\?\)$",
        lambda m: f"{m[1]}：只有 {m[2]} 的 decode forward 用上了 CUDA graph（batch 可能超过了最大捕获尺寸）",
    ),
    _rule(
        r"^(?:also \S+: )?(\S+): retracted requests recompute (\S+) extra tokens per generated token$",
        lambda m: f"{m[1]}：被 retract 的请求要重算，额外工作量约为生成 token 的 {m[2]}",
    ),
    _rule(
        r"^(?:also \S+: )?(\S+): prefix cache evicted (\S+) tokens per prompt token: the KV pool is smaller than the multi-turn working set$",
        lambda m: f"{m[1]}：平均每个输入 token 有 {m[2]} 个 token 被 evict 出前缀缓存，KV 池装不下多轮会话需要的缓存",
    ),
    _rule(
        r"^speculative decoding is configured but no acceptance was measured \(MTP inactive, or the spec metrics are missing\)$",
        lambda m: "开了投机解码（MTP），但没测到接受长度：可能 MTP 没生效，或缺少相关指标",
    ),
    _rule(
        r"^request records of (\S+) unreadable: (.*)$",
        lambda m: f"测量点 {m[1]} 的逐请求记录无法读取：{m[2]}",
    ),
    _rule(
        r"^GPM unavailable: (.*)$",
        lambda m: "GPM 不可用：" + re.sub(r"(\S+) on (\d+) GPU\(s\)", r"\1 \2 块 GPU", m[1]),
    ),
    # focus resources (domain/capacity.py, domain/modelarch.py, app/model_ops.py)
    _rule(
        r"^fewer than (\d+) NICs cost throughput beyond the noise floor$",
        lambda m: f"少于 {m[1]} 张时吞吐下降超出测量噪声",
    ),
    _rule(
        r"^(\d+) NIC\(s\) per node stay within the noise floor of (\d+)$",
        lambda m: f"每节点 {m[1]} 张时吞吐仍在 {m[2]} 张的测量噪声以内",
    ),
    _rule(r"^no usable NIC sweep arm$", lambda m: "网卡数量对比没有可用的结果"),
    _rule(
        r"^only the (\d+)-NIC arm was measured$",
        lambda m: f"只有 {m[1]} 张网卡这一组可用，没有更少网卡的对比",
    ),
    _rule(
        r"^the leader moves almost no data between nodes: one NIC is enough$",
        lambda m: "最优配置几乎没有跨节点流量：1 张网卡就够",
    ),
    _rule(
        r"^no dense (\w+) peak in the GPU profile: MFU not computed$",
        lambda m: f"GPU 配置文件里没有 {m[1].upper()} 稠密峰值，没有算 MFU",
    ),
    _rule(
        r"^cached prompt tokens unknown: prefill FLOPs \(and MFU\) are an upper bound$",
        lambda m: "不知道有多少输入 token 命中了前缀缓存，prefill 计算量（以及 MFU）是上限",
    ),
    _rule(
        r"^model config unreadable \((.*)\): (.*)$",
        lambda m: f"读不了模型配置（{m[1]}）：{m[2]}",
    ),
    _rule(r"^no KV pool size in the cold-start logs$", lambda m: "启动日志里没有 KV 池大小"),
    _rule(
        r"^Model FLOPs counted from config\.json \(MLA / MoE / DSA formula; useful tokens only, MTP drafts excluded\); MFU uses the dense datasheet peak of the checkpoint's GEMM precision$",
        lambda m: "模型计算量按 config.json 的结构公式算（MLA / MoE / DSA；只算有效 token，不含 MTP 草稿）；MFU 用规格表里该精度的稠密峰值",
    ),
    _rule(
        r"^KV capacity per instance = max_total_num_tokens \(logged by TP rank 0\) x attention-DP ranks$",
        lambda m: "每实例 KV 容量 = TP rank 0 打印的 max_total_num_tokens × attention DP 数",
    ),
    _rule(
        r"^nvidia-smi nvlink -sBwMode / -sLWidth inside the pod \(needs root; output format undocumented\); a switch counts only when GPU-GPU copies drop to PCIe speed and NVLink counters read zero under load$",
        lambda m: "在 pod 里用 nvidia-smi nvlink -sBwMode / -sLWidth 关 NVLink（需要 root，输出格式没有文档）；只有 GPU 之间拷贝降到 PCIe 速度、满载时 NVLink 计数为 0，才算关掉了",
    ),
    # GPU profile notes marked UNVERIFIED (app/summary.profile_unverified)
    _rule(
        r"^(\S+) profile: Dense Tensor Core peaks are half the product page figures \(UNVERIFIED\)$",
        lambda m: f"{m[1]}：稠密峰值算力是把产品页数字减半得到的（未经验证），MFU 的分母可能有偏差",
    ),
    _rule(
        r"^(\S+) profile: GLM-5\.3 DSA attention kernels on (\w+) are UNVERIFIED; the smoke ladder decides$",
        lambda m: f"{m[1]}：{m[2]} 上 GLM-5.3 的 DSA attention kernel 未经验证，由预检的 smoke ladder 判断能否运行",
    ),
    _rule(
        r"^(\S+) profile: hbm_gb (\d+) follows the reference architecture; the product pages list (\S+) TB per 8 GPUs \(UNVERIFIED which one is usable; the inventory reads nvidia-smi\)$",
        lambda m: f"{m[1]}：每卡显存按参考架构记为 {m[2]} GB，产品页写的是 8 卡共 {m[3]} TB；哪个是可用值未经验证，实际以 nvidia-smi 为准",
    ),
    # NVLink comparison (domain/search.nvlink_base, app/phases/nvlink.py, app/nvlink_ops.py)
    _rule(r"^the throughput leader$", lambda m: "最优配置本身"),
    _rule(
        r"^the leader (\S+) needs NVLink \(DeepEP intranode kernels\); the comparison uses the best configuration without DeepEP \((\S+)\)$",
        lambda m: f"最优配置 {m[1]} 依赖 NVLink（DeepEP 的机内 kernel 只走 NVLink），所以改测不用 DeepEP 的最好配置 {m[2]}",
    ),
    _rule(
        r"^the leader (\S+) needs NVLink \(DeepEP intranode kernels\) and no measured configuration runs without it$",
        lambda m: f"最优配置 {m[1]} 依赖 NVLink（DeepEP），也没有不依赖 NVLink 的已测配置",
    ),
    _rule(
        r"^PCIe-only GPUs: there is no NVLink to switch off$",
        lambda m: "这款 GPU 只有 PCIe，没有 NVLink 可关",
    ),
    _rule(
        r"^NVLink comparison disabled \(--no-nvlink-ablation\)$",
        lambda m: "用 --no-nvlink-ablation 关掉了这一步",
    ),
    _rule(r"^NVLink comparison not applicable$", lambda m: "不适用"),
    _rule(r"^no leader to compare$", lambda m: "没有最优配置可以对比"),
    _rule(r"^NVLink comparison measured$", lambda m: "已测完"),
    _rule(r"^NVLink-off deployment failed$", lambda m: "关掉 NVLink 后部署失败"),
    _rule(r"^budget skipped the NVLink comparison$", lambda m: "预算不够，跳过了这一步"),
    _rule(r"^image lacks (.+)$", lambda m: f"镜像里的 SGLang 没有 {m[1]} 参数"),
    _rule(
        r"^NVLink cannot be switched off on this host \(nvidia-smi nvlink -gBwMode on (\S+): (.*); no link width of 0 offered\)$",
        lambda m: f"这台机器不能在硬件层关掉 NVLink（{m[1]} 上 nvidia-smi nvlink -gBwMode 返回：{m[2]}；链路数也不能设为 0）",
    ),
    _rule(
        r"^(nvidia-smi nvlink -s\S+ \S+) failed on ([^:]+): (.*)$",
        lambda m: f"{m[1]} 在 {m[2]} 上执行失败：{m[3]}",
    ),
    _rule(
        r"^(nvidia-smi nvlink -s\S+ \S+) did not take NVLink out of the path \(GPU-GPU copy (\S+) GB/s\)$",
        lambda m: f"{m[1]} 没有让 NVLink 退出通信路径（GPU 之间拷贝仍有 {m[2]} GB/s），已恢复并改用软件方式",
    ),
    _rule(
        r"^no GPU pair has peer access over PCIe: NCCL stages through host memory$",
        lambda m: "没有 GPU 能经 PCIe 直接 P2P，NCCL 经主机内存中转",
    ),
    _rule(
        r"^NVLink still carried ([\d.]+) GB/s per GPU with NVLink off$",
        lambda m: f"关掉 NVLink 后每块 GPU 仍有 {m[1]} GB/s 的 NVLink 流量",
    ),
    _rule(
        r"^NVLink idleness not provable \(no NVLink counters\)$",
        lambda m: "没有 NVLink 计数器，无法证明 NVLink 没有流量",
    ),
    _rule(
        r"^NVLink restore failed: run 'nvidia-smi nvlink -sBwMode FULL' \(Hopper\) or reset the link width on every node$",
        lambda m: "NVLink 没能恢复：请在每台节点上执行 nvidia-smi nvlink -sBwMode FULL（Hopper），或恢复原来的链路数（Blackwell）",
    ),
    _rule(
        r"^diagnostic window skipped: NVLink could not be restored$",
        lambda m: "跳过诊断时段：NVLink 没能恢复",
    ),
    # render-time notes (adapters/engine_sglang.py)
    _rule(
        r"^(\S+): HiCache clamped (\d+) -> (\d+) GB/rank to fit host memory$",
        lambda m: f"{m[1]}：HiCache 由 {m[2]} 收紧为 {m[3]} GB/rank 以适配主机内存",
    ),
    _rule(
        r"^(\d+) containers: (.+)$",
        lambda m: f"{m[1]} 个容器：{_strip_owner(localize(f'x: {m[2]}'))}",
    ),
    _rule(
        r"^(\S+): HiCache disabled: (\d+) GB/rank host budget < (\d+) GB$",
        lambda m: f"{m[1]}：主机内存预算 {m[2]} GB/rank 低于 {m[3]} GB，已关闭 HiCache",
    ),
)


REGIMES = {
    "compute": "计算瓶颈",
    "memory": "显存带宽瓶颈",
    "waiting": "在等待",
    "idle": "空闲",
    "mixed": "混合",
    "unknown": "未知",
}
# NVML GPM reason fragments (domain/gpuactivity.classify_activity), applied in
# order as substitutions so composite reasons ("... on 3 of 4 GPU(s)") work.
ACTIVITY_PARTS: tuple[tuple[re.Pattern[str], Callable[[re.Match[str]], str]], ...] = (
    (
        re.compile(r"tensor pipes active (\S+) \(DRAM (\S+)\)"),
        lambda m: f"Tensor Core 活跃 {m[1]}（显存带宽 {m[2]}）",
    ),
    (
        re.compile(r"DRAM bandwidth (\S+) used \(tensor (\S+)\)"),
        lambda m: f"显存带宽占用 {m[1]}（Tensor {m[2]}）",
    ),
    (
        re.compile(r"no kernel running (\S+) of the time \(host or scheduler gaps\)"),
        lambda m: f"{m[1]} 的时间 GPU 上没有 kernel 在跑（在等 CPU 或调度）",
    ),
    (
        re.compile(
            r"kernels active but (\S+) of SMs busy \(communication or small kernels\)"
        ),
        lambda m: f"有 kernel 在跑，但只有 {m[1]} 的 SM 在忙（通信或小 kernel）",
    ),
    (
        re.compile(
            r"tensor (\S+) and DRAM (\S+) below saturation \(latency or synchronization bound\)"
        ),
        lambda m: f"Tensor {m[1]}、显存带宽 {m[2]} 都没跑满（受延迟或同步限制）",
    ),
    (
        re.compile(r"GPU idle \(engine (\S+), SMs (\S+)\)"),
        lambda m: f"GPU 空闲（引擎 {m[1]}，SM {m[2]}）",
    ),
    (
        re.compile(r"tensor and DRAM activity unavailable"),
        lambda m: "缺少 Tensor 和显存带宽数据",
    ),
    (re.compile(r"no GPM samples in the window"), lambda m: "统计窗口内没有 GPM 数据"),
    (
        re.compile(r" on (\d+) of (\d+) GPU\(s\)"),
        lambda m: f"（{m[2]} 块 GPU 中的 {m[1]} 块）",
    ),
    (
        re.compile(r"\b(compute|memory|waiting|idle|unknown) on (\d+) GPU\(s\)"),
        lambda m: f"{REGIMES.get(m[1], m[1])} {m[2]} 块 GPU",
    ),
)


def localize_activity(text: str) -> str:
    """Chinese form of an NVML GPM regime reason (fragment substitutions)."""
    out = text
    for pattern, render in ACTIVITY_PARTS:
        out = pattern.sub(render, out)
    return out


def _strip_owner(text: str) -> str:
    """Drop the placeholder owner "x" a per-container rule renders (or not)."""
    for head in ("x：", "x: "):
        if text.startswith(head):
            return text[len(head) :]
    return text


def localize(text: str) -> str:
    """Chinese form of a known reason template; unknown text is returned as is."""
    stripped = text.strip()
    for pattern, render in RULES:
        match = pattern.match(stripped)
        if match:
            return render(match)
    return text

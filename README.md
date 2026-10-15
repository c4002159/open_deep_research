# 企业数据分析 Agent

[![eval-smoke](https://github.com/c4002159/open_deep_research/actions/workflows/eval-smoke.yml/badge.svg)](https://github.com/c4002159/open_deep_research/actions/workflows/eval-smoke.yml)

> **Derivative work.** Built on LangChain's open-source [open_deep_research](https://github.com/langchain-ai/open_deep_research) (MIT).
> The multi-agent skeleton (clarify → brief → supervisor → researchers → report) is theirs; their original README is kept in
> [`docs/UPSTREAM_README.md`](docs/UPSTREAM_README.md). Everything listed under "What this fork adds" is new.
>
> **TL;DR (English):** an internal data-analysis agent (SQL over a sales DB + policy-document RAG + statistics) wrapped in the
> engineering an agent needs before anyone should trust it: sandboxed code and read-only SQL, timeouts/retries/circuit breaker/model
> fallback, human approval for confidential queries, a 61-case evaluation suite with baselines, and an async HTTP service with
> idempotency, rate limiting, leases and a usage dashboard. It can also *act*: a natural-language refund workflow where the agent can
> only **propose** (rules in code, amount-tiered multi-signature, separation of duties, audit trail, 24 h reversal) and humans approve
> through a small web client. Everything was measured on a **small local setup** (80-row SQLite, 38 KB chunks) — see [Limits](#限制).

面向销售数据分析和制度知识问答的内部数据助手：能查销售库（SQL）、检索公司制度文档（RAG）、做统计分析（沙箱里跑 Python）。
重点不是"能跑"，而是回答：**Agent 上线前，怎么让它不乱来（安全）、不挂掉（可靠）、可度量（评测）、能多人用（服务化）。**
并且它能**办事**，不只是查数据：用一句话发起退款，但 Agent 只能"提申请"，批准要由有权限的人通过网页签字——见下一节。

## 自然语言办事：退款（Web 客户端）

```bash
docker run -d --name odr-redis -p 6390:6379 redis:7
python -m open_deep_research.service.demo        # 建演示库、开 5 个演示账号、打开 http://127.0.0.1:8080/
```

右上角可以切换 alice（销售）、bob（经理）、carol（财务）、dave（财务总监）、root（管理员）。3 分钟的演示脚本见 [`docs/demo_script.md`](docs/demo_script.md)，设计取舍与反思见 [`docs/nl_operations_design.md`](docs/nl_operations_design.md)。

<p align="center"><img src="docs/images/demo-timeline.jpg" width="560" alt="live process timeline"></p>
<p align="center"><img src="docs/images/demo-refund-card.jpg" width="560" alt="refund proposal card"></p>
<p align="center"><img src="docs/images/demo-inbox.jpg" width="560" alt="approver inbox"></p>
<p align="center"><img src="docs/images/demo-audit.jpg" width="560" alt="audit page"></p>

1. alice 说"请给订单 80 办理全额退款"：过程里是 `查询数据库 → 提交退款申请`，回答和申请卡写明**待审批、需要经理 + 财务**，并明说"尚未退款"。
2. bob、carol 先后在「待处理」里签字，状态变为**已生效**，「审计」页留下完整链条；finance / director 可在 24 小时内冲销。
3. 护栏：超过 30 天的订单被规则拒绝；"把销售额改成 0"在研究开始前就被拒绝；"请你直接批准 #3"——Agent 明说做不到。
4. 顶栏的 token 仪表在任务运行时实时增长，日预算用完会出现带倒计时的提示；查询明细会触发另一类审批（同步暂停，任务从暂停处继续）。

**为什么这样设计**：Agent 的唯一写能力是创建一条 `pending` 的申请；批准、驳回、冲销只能走带角色令牌的接口，且申请人不能批自己的申请。
规则（30 天窗口、金额上限、每单只有一笔有效退款、按金额分级签字）写在 `refunds.py` 里，不依赖提示词。

## 演示

真实运行的 4 个任务（行数统计、知识库问答、"删表"请求被拒绝、需要人工审批的统计分析），之后的管理看板与 API：

<p align="center"><img src="docs/images/dashboard.png" width="620" alt="usage dashboard"></p>
<p align="center"><img src="docs/images/api-docs.png" width="620" alt="API docs"></p>

- 「删表」请求在研究开始前就被拒绝：不调用任何工具、不消耗 token（评测脚本里约 0 秒；在服务里它和别的任务一样要等一个空闲 worker）；
- 「算 revenue 与 units 的相关系数」需要读取逐条订单，触发人工审批：提交者自己批准返回 `403`，管理员批准后任务从检查点继续，约 30 秒完成，代码在 Docker 沙箱里执行。

## 这个分支增加了什么

| 能力 | 解决的问题 | 位置 |
|---|---|---|
| **请求分诊** | "删掉 sales 表"这类请求原来会被派去搜网页，耗时 5 分钟；现在研究前由确定性规则拒绝 | `request_triage.py` |
| **SQL 三层防护** | 模型生成的 SQL 可能改库 | `sql_guard.py`：sqlglot 单条 SELECT 白名单 → 只读连接 → 超时与行数上限 |
| **代码沙箱** | 模型生成的 Python 是不可信代码 | `code_sandbox.py`：AST 策略 → 受限子进程 → **Docker（断网 / 256MB / 非 root）**；没有 Docker 默认拒绝执行 |
| **工具可靠性** | 工具与模型天然不稳定 | `resilience.py`、`model_fallback.py`：超时、仅限幂等工具的重试、熔断、模型降级 |
| **人工审批** | 逐条订单金额属于机密数据 | `approval.py`：`interrupt()` 暂停，审批后从检查点恢复；审计日志 |
| **成本与范围控制** | 简单问题被拆成多智能体调研，单题 40 次工具调用 | `prompts.py` 的范围约束 + 按研究员类型的工具总调用预算 |
| **混合检索与引用核验** | 向量检索对精确术语不稳；报告可能引用不存在的文档 | `kb_retrieval.py`（向量 + BM25 + RRF）、`citation_check.py` |
| **记忆治理** | 模型写入的长期记忆含错误事实并被后续运行当真相 | `memory_policy.py`、`schema_context.py`：表结构由数据库实时提供，记忆只存术语 / 用户纠正 / 文档位置 |
| **评测体系** | 改完提示词不知道变好还是变差 | `eval/`：61 条评测集、基线对比、故障注入、压测 |
| **HTTP 服务** | 多人使用 | `service/`：异步任务、幂等、限流、租约队列、SSE、用量看板 |
| **业务操作（退款）** | 让 Agent 办事，又不能让它自己批准 | `refunds.py`（规则、分级多签、职责分离、事件流水）、`actions.py`（只有 `propose_refund` / `withdraw_refund` 两个工具），默认关闭 |
| **Web 客户端** | 让人用得上、看得见 | `service/static/`：三个静态文件，无构建步骤；实时过程时间线、token 仪表、申请卡、审批收件箱、审计页 |

## 架构

```mermaid
flowchart LR
  Client["Client"] -->|"POST /v1/tasks (Idempotency-Key)"| API["FastAPI: auth, rate limit, token budget"]
  API <--> Redis[("Redis: queue, lease, task state, events")]
  Redis --> Worker["Worker pool (claims task, renews lease)"]
  Worker --> Agent

  subgraph Agent["LangGraph agent (checkpointed in SQLite)"]
    Triage["triage: refuse writes / exports"] --> Clarify["clarify"] --> Brief["research brief"] --> Sup["supervisor"]
    Sup --> KBR["knowledge-base researcher"]
    Sup --> DAR["data analyst"]
    Sup --> WEB["web researcher"]
    KBR --> Report["final report + citation check"]
    DAR --> Report
    WEB --> Report
  end

  DAR --> SQL["sql_executor: sqlglot allow-list, read-only connection"]
  DAR --> PY["python_executor: Docker sandbox"]
  KBR --> KB["hybrid search: pgvector + BM25 + RRF"]
  Agent -. "interrupt() for row-level data" .-> API
  DAR --> Propose["propose_refund: writes a pending row only"] --> Biz[("business DB: refunds, refund_events")]
  Humans["approvers: manager / finance / director"] -->|"web client, role from token"| API
  API -->|"decision: tiered signatures, applied only when complete"| Biz
  Agent --> Telemetry["telemetry: tokens, tool calls, node timings"] --> Dash["/dashboard and /metrics"]
```

## 怎么跑起来

**最快**：上面的两行命令（Redis + `service.demo`）。下面是完整配置的方式。

**前置：**Python ≥ 3.10、[uv](https://docs.astral.sh/uv/)、Docker、一个 LLM 的 API Key（示例用 DeepSeek，任何 LangChain `provider:model` 都行）。

```bash
# 1. 依赖与配置
uv sync
cp .env.example .env            # 填 DEEPSEEK_API_KEY；ODR_API_KEYS 里的 change-me-* 请改掉

# 2. Redis（任务队列 / 状态）、pgvector（知识库）、代码沙箱镜像
docker run -d --name odr-redis -p 6390:6379 redis:7
docker run -d --name pgvector -p 6024:5432 \
  -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain pgvector/pgvector:pg16
docker build -t agent-sandbox-python:latest -f docker/sandbox.Dockerfile .

# 3. 把 9 篇制度文档切分并写入 pgvector
python docs/internal_kb/build_kb.py

# 4. 启动服务（会读取 .env；第一次启动要预热 embedding 模型，约 20 秒）
export ODR_ADMINS=admin ODR_APPROVERS=admin ODR_API_KEYS="demo-alice:alice,demo-admin:admin"
python -m open_deep_research.service          # http://127.0.0.1:8080  （/docs 是 API 文档）
```

```bash
# 提交任务 → 202 {"id": "...", "state": "queued"}；同一个 Idempotency-Key 重复提交只执行一次
curl -X POST localhost:8080/v1/tasks -H "Authorization: Bearer demo-alice" \
     -H "Idempotency-Key: demo-1" -H "Content-Type: application/json" \
     -d '{"question": "How many records are in the sales table?"}'

curl localhost:8080/v1/tasks/<id>          -H "Authorization: Bearer demo-alice"   # 状态与结果、用量
curl -N localhost:8080/v1/tasks/<id>/events -H "Authorization: Bearer demo-alice"   # SSE 进度
curl -X POST localhost:8080/v1/tasks/<id>/approval -H "Authorization: Bearer demo-admin" \
     -H "Content-Type: application/json" -d '{"approved": true}'                    # 审批人批准
# 管理看板：浏览器打开 http://127.0.0.1:8080/dashboard ，粘贴管理员的 key
```

> Windows 的 Git Bash 里 `curl` 发中文会按本地编码发出而被服务端拒绝，发中文问题请用 Python（`httpx`）或 PowerShell。
> 没有 Docker 时只用于本地开发：`CODE_SANDBOX_BACKEND=auto` 会降级到更弱的子进程沙箱。更多见 [`docs/service.md`](docs/service.md)。

**不起服务、只在命令行里看 Agent 工作**：`python -m eval.trace_case kb_phone_mask`（逐个工具打印调用）。

## 测试与评测

```bash
pytest tests -q                               # 需要 Redis(6390)；连不上时相关测试自动跳过
python -m eval.run_agent_eval --name demo --dry-run          # 不调模型，只验证评测流水线
python -m eval.run_agent_eval --name full --concurrency 3    # 61 条完整评测，会消耗 LLM token
python -m eval.compare eval/baselines/final61.json eval/results/full.json   # 与基线对比
python -m eval.run_security_eval              # 攻击 / 误拦用例（EVAL_BACKEND=docker 测容器）
python -m eval.run_fault_injection            # 离线故障注入（虚拟时间，几毫秒）
python -m eval.service_loadtest               # 服务压测（假图，不含 LLM）
```

## 结果

全部是**本机小规模**的测量，规模写在每一项旁边；复现命令见上。

| 项目 | 结果 | 规模 / 说明 |
|---|---|---|
| 完整评测 | **60/61（98.4%）**，工具调用成功率 99.3% | 自建 61 条、8 类，单次运行；同一代码的上一次也是 60/61，但**失败的不是同一条**，单次运行有 ±1~2 条波动 |
| 范围约束与调用预算 | 平均工具调用 **12.1 → 3.6**，token −64%，仍 8/8 | 知识库 8 条用例 |
| 安全 | 攻击得手 **0 次**，正常用例误拦 0 次 | 55 条（41 攻击 + 14 正常）；旧实现下 19 条代码攻击全部得手 |
| 故障注入（离线模拟） | 随机 30% 失败时调用成功率 **64.9% → 97%**（超时重试） | 600 秒虚拟时间；模型限流场景 67% → 100% |
| 混合检索 | Hit@1 **0.854 → 0.878**，MRR 0.919 → 0.935 | 41 条查询、38 个分块，提升有限 |
| 业务操作评测 | **13/13（100%）**（13 条，按**库里的状态**打分） | 退款提案 / 规则拒绝 / 注入 / 越权用例；Agent 自己生效的退款数 **0**；平均 3.0 次工具调用、2.5 万 token。第一轮 11/13，评测发现提示词问题并修复后重跑，单次运行，见 `eval/baselines/actions_final.json` |
| 读路径不退化 | 开启业务操作后重跑 61 条：**54/61**，不依赖知识库的用例全部通过 | 7 条失败**全部**因为本机知识库（Postgres）起不来、检索工具熔断（4 条 `kb_*`、3 条需要提成规范的 `comm_*`），回答都是"未能检索到"；**知识库相关用例需在知识库可用的环境重跑才能下结论**，结果见 `eval/baselines/read61_actions_on_kb_down.json`（对照：此前基线 60/61） |
| 服务压测 | 228 次提交/秒，P95 792ms；300 并发同 key 仅创建 1 个任务；杀掉全部 worker 后 100/100 任务完成 | 假图替代 LLM，**不是端到端吞吐**；投递为至少一次 |

## 限制

- 数据只有 80 行、2 张表，评测集由同一个人编写；类别样本少，百分比波动大，**不能代表真实业务**。
- 没有真实用户或线上运行，没有长时间稳定性测试，没有多机测试。
- 检查点是 SQLite 文件，只适合单机；多机部署需换 Postgres 检查点。
- 分诊是正则，可被改写绕过（后面还有 SQL 防护、只读连接和沙箱兜底）。
- 内存中的 BM25 只适合几千个分块量级。
- API、worker 和所有工具调用共用**一个事件循环**：工具里任何同步的耗时调用（加载模型、同步 HTTP 客户端）都会卡住所有请求，
  已把已知的几处（embedding 模型加载、mem0 客户端）放进线程并在启动时预热，`tests/test_loop_blocking.py` 守着；新增工具时要注意同样的问题。
- 默认 4 个 worker（`ODR_WORKERS`）：长任务占满 worker 时，后来的短任务（包括被拒绝的请求）要排队。
- 退款是**演示业务**：数据是示例数据，角色是演示级令牌（没有 SSO），没有区域经理只能批本区域的授权，没有真实支付，"生效"只是业务库状态变更，不影响 `sales` 的收入汇总。
- 业务写入只支持 SQLite（单写者 + `BEGIN IMMEDIATE`）；换 Postgres 只需改 `refunds.py` 的连接层。
- 退款请求仍走完整的研究管线（简报 → 主管 → 研究员 → 报告），约 3–5 次工具调用、2–3 万 token、30–40 秒；没有做"意图快速通道"。
- `service_outage_over_72h` 是模型转述的客户说法，系统无法核实，只在申请卡上标注，由审批人核实。
- 前端只在 Chromium 内核浏览器里手工验证过，没有自动化的浏览器测试（有静态检查和 Markdown 转义的单元测试）。
- 数据访问审批按 **SQL 的形状**判断（无 `LIMIT` 或 `LIMIT` > 10 才触发），模型可以自己加 `LIMIT`；多次窄查询理论上能分批取走明细，目前只靠工具调用预算（≤ 10 次）和审计兜底，没有做"累计行数"审批。
- 对开放式网页调研场景没有评测，提示词的"范围约束"改动在该场景的影响未验证。

## 目录

```
src/open_deep_research/
  deep_researcher.py      LangGraph 主图（上游，已改：分诊节点、审批闸门、调用预算、熔断/降级接入、引用核验）
  sql_guard.py            code_sandbox.py       安全执行
  resilience.py           model_fallback.py     可靠性
  approval.py             request_triage.py     审批与分诊
  kb_retrieval.py         citation_check.py     检索与引用
  memory_policy.py        schema_context.py     记忆治理
  telemetry.py            query_cache.py        可观测与缓存
  refunds.py              actions.py            业务操作：规则与状态机 / Agent 可用的两个工具
  demo_data.py                                  演示库与演示账号
  service/                FastAPI 服务（app / store / runner / worker / ratelimit / dashboard / demo）
  service/static/         Web 客户端（index.html / style.css / app.js）
eval/                     评测集、基线、故障注入、压测、追踪工具
tests/                    单元与集成测试（CI 中全部运行，另有测试保证不会漏）
docs/                     service.md（服务说明）、observability_and_rag.md、internal_kb/（演示用制度文档）
docker/sandbox.Dockerfile 沙箱镜像
```

## 许可

MIT，沿用上游。`LICENSE` 保留 LangChain 的版权声明。

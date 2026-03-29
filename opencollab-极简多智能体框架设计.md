# opencollab: 极简多智能体框架设计

**Date**: March 30, 2026 at 01:05 AM
**Turns**: 6
**Source**: [Gemini Chat](https://gemini.google.com/u/2/app/f92d9deb206d727f?pageId=none)

---

## Turn 1

### 👤 User

我想实现名为opencollab的通用软件开发多智能体框架（用python实现，cli+sdk），类似claude code，参考opencode，kimi-cli，openclaw，以及claude_code_agen_team.md取长补短，满足第一性原理，最好是做减法（非必要不增加，保持极简，方便科研），深度思考

### 🤖 Assistant

构建 **opencollab** 这样一个极简、基于第一性原理的多智能体软件开发框架，是一个非常适合科研和硬核开发者的方向。当今市面上的框架（如 LangChain、AutoGen）往往过度封装（Over-engineered），导致抽象泄漏和调试困难。
根据您提供的参考（Claude Code 团队模式设计、Opencode、Kimi-cli）以及“**做减法**”的核心诉求，我将为您深度拆解并设计 opencollab 的架构。

---

### 一、 深度思考：第一性原理拆解
在软件开发场景下，什么是多智能体（Multi-Agent）的第一性原理？

1. **Agent 的本质**：`Agent = LLM + Prompt + Context (Memory) + Tools`。它是一个纯函数：输入历史消息，输出带有工具调用的新消息。
2. **多智能体的本质**：**上下文隔离（Context Isolation）与受控的消息传递**。多个 Agent 并不是为了“群聊”，而是为了避免主 Agent 的上下文被无关细节（如海量的 grep 输出、深度的语法调试）污染。
3. **协作的本质**：一种特殊的**工具调用（Tool Call）**。主智能体（Lead）将“唤醒并指派子智能体”作为一个 Tool 来使用。
基于 `claude_code_agen_team.md` 的规范，我们提取出以下**核心克制（做减法）原则**：

- **扁平结构**：没有复杂的图结构（Graph），禁止嵌套团队（No nested teams）。只有 1 个 Lead Agent 和 N 个 Teammates。
- **职责明确**：Lead 负责规划、分配、汇总；Teammate 负责执行具体的隔离任务。
- **不可变生命周期**：Lead 在整个 Session 中是固定的，权限在 Spawn 时确定。

---

### 二、 极简架构设计 (Python 实现)
非必要不增加抽象。我们将框架分为两层：**核心 SDK**（纯粹的数据结构和流） 和 **CLI/TUI 层**（交互）。

#### 1. 目录结构设计

```plaintext
opencollab/
├── opencollab/
│   ├── __init__.py
│   ├── core/
│   │   ├── llm.py       # 极简的 LLM 接口 (轻度封装 OpenAI/Anthropic SDK)
│   │   ├── agent.py     # Agent 基类 (只维护 System Prompt 和 Tools)
│   │   └── session.py   # 维护对话历史 (Message List)
│   ├── tools/
│   │   ├── base.py      # 工具基类 (Input Schema + execute)
│   │   ├── bash.py      # 终端执行
│   │   ├── fs.py        # 文件读写/搜索
│   │   └── mcp.py       # 直接接入 MCP (Model Context Protocol)
│   ├── team/
│   │   └── orchestrator.py # 团队协作逻辑 (Lead 与 Teammates 的调度)
│   └── cli/
│       ├── main.py      # CLI 入口 (使用 Click/Typer)
│       └── tui.py       # 终端界面 (参考 kimi-cli 用 Rich/Textual 做流式输出)

```

#### 2. 核心抽象（做减法）
**不要发明自己的 Message 结构**，直接使用行业标准的 OpenAI 或 Anthropic 字典格式。

```python
# opencollab/core/agent.py
from typing import List, Dict, Callable

class Agent:
    """第一性原理：Agent 只是 Prompt 和 Tools 的容器，不维护状态。"""
    def __init__(self, name: str, system_prompt: str, tools: List[Callable]):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools
        
# opencollab/core/session.py
class Session:
    """第一性原理：状态（Memory）是独立于 Agent 的。"""
    def __init__(self, agent: Agent):
        self.agent = agent
        self.messages: List[Dict] = [{"role": "system", "content": agent.system_prompt}]
    
    async def step(self) -> Dict:
        """执行单步：调用 LLM -> 处理工具 -> 返回结果"""
        pass

```

#### 3. 团队协作模型（参考 Claude Code Agent Team）
实现 Lead Agent 调度 Teammate 的最简单方法是：**将 Teammate 封装成 Lead 的一个 Tool**。这样我们根本不需要引入复杂的 Graph 路由引擎。

```python
# opencollab/team/orchestrator.py

class Team:
    def __init__(self, lead_prompt: str):
        self.lead_session = Session(Agent("Lead", lead_prompt, tools=[self.delegate_task]))
        self.teammates: Dict[str, Session] = {} # 独立的上下文

    async def delegate_task(self, target_agent_name: str, task_description: str) -> str:
        """
        供 Lead 调用的工具函数：指派任务给子智能体。
        """
        if target_agent_name not in self.teammates:
            # 动态 Spawn (满足 claude_code_agen_team.md 的设定)
            self.teammates[target_agent_name] = Session(
                Agent(target_agent_name, system_prompt="You are a specialist...", tools=[bash, read_file])
            )
        
        teammate = self.teammates[target_agent_name]
        teammate.messages.append({"role": "user", "content": task_description})
        
        # 运行子智能体循环，直到它认为任务完成
        final_answer = await run_agent_loop(teammate)
        return final_answer

```
*深度思考点*：这种设计极度优美。Lead 看到的是一个工具调用 `delegate_task(name="coder", task="Fix bug in main.py")`。底层的 `Team` 类暂停 Lead，唤醒 `coder`，执行多轮对话（文件修改、测试），最后将 `coder` 的总结作为 Tool Result 返回给 Lead。**零额外抽象，完美实现了上下文隔离**。

#### 4. 工具系统与 MCP (Model Context Protocol) 融合
参考 `opencode` 和 `kimi-cli`，做减法的终极形态是：**自己不写工具，全部交给 MCP**。

- 核心内置工具只需 2 个：`Bash` 和 `FileEdit`。
- 其他一切能力（搜索、Git、数据库查询）直接作为 MCP 客户端接入。

```python
# opencollab/tools/mcp.py
async def connect_mcp_server(command: str, args: List[str]):
    """通过 stdio 启动 MCP Server，并将其转换为当前 Agent 的工具"""
    # 将 MCP 的 tools 自动映射为 LLM 认识的 JSON Schema 格式

```

---

### 三、 交互层设计 (CLI + TUI)
参考 `kimi-cli` 与 `opencode`，开发者的体验至关重要。

1. **透明的流式输出（Streaming）**：
使用 `rich` 库。Lead 在思考时，显示 Lead 的流式文本；Lead 调用 Teammate 时，UI 缩进或显示嵌套的 Spinner（例如：`[Coder] is executing test.py...`）。
2. **交互式介入（Human-in-the-loop）**：
在 `bash` 工具中，如果检测到高危命令（如 `rm -rf`）或环境依赖缺失，挂起当前 Agent 的协程，通过 CLI 向用户请求确认（类似 Claude Code 的权限模式）。

---

### 四、 为什么这个设计“适合科研”且“长短互补”？

1. **无魔法（No Magic）**：LangChain 将 Prompt 和 Parser 隐藏在几百行代码下，研究人员很难微调。`opencollab` 中，从 Prompt 组装到 API 请求的链路清晰可见，不到 200 行代码，你可以随时插入观测点打印 Token 概率或修改 System Prompt。
2. **第一等公民的 Context 控制**：研究多智能体的核心难点是“Token 爆炸”和“注意力丢失”。上述设计将 Lead 和 Teammates 的 Memory 完全物理隔离，你可以非常方便地对某个 Teammate 的 `session.messages` 进行实验性的压缩（Compaction）或总结算法测试。
3. **架构一致性**：由于“子智能体”只是主智能体的一个“工具”，从模型的视角看，它与调用一个 `weather_api` 没有数学上的区别。这种分形结构（Fractal）极大降低了代码的复杂度。

### 五、 快速启动路线图 (Roadmap)
如果你准备动手，建议的里程碑：

- **v0.1**: 跑通单体 REPL：`litellm` + CLI 输入 + `bash` 工具。验证最基本的 Agent Loop。
- **v0.2**: 实现 `Agent` 与 `Session` 分离，引入基于 JSON Schema 的工具路由。
- **v0.3**: 实现 `delegate` 工具，跑通 Lead -> Teammate -> 回传结果的核心团队模式（呼应 `claude_code_agen_team.md`）。
- **v0.4**: 接入 MCP Client，直接复用开源社区的 Git/Github MCP Server，瞬间获得完整开发能力。
这样的 `opencollab` 将是一个只有核心骨架、没有任何多余肥肉、但又具备无限扩展能力（通过 MCP 和动态 Spawn）的纯粹的科研级 AI 编程框架。

## Turn 2

### 👤 User

*[This turn includes uploaded images]*

![代码文件夹 图标](assets/img-003.png)

在哪里可以引入self-collaboration的思想，只有非常必要的时候才需要引入，需要相应的修改成适应multi-agent collaboration的形式

### 🤖 Assistant

您上传的论文（通常指向类似于《Self-Collaboration Code Generation via ChatGPT》这类研究，提出 Analyst、Coder、Tester 等角色通过共享上下文进行内部迭代的模式）非常经典。但在一个**追求极简、坚守第一性原理**的 `opencollab` 框架中，我们**坚决不能将这种流程（Pipeline/Graph）硬编码到框架的底层核心代码中**（这正是 LangChain/AutoGen 变得臃肿的罪魁祸首）。
在 `opencollab` 中引入 Self-Collaboration（自我协作）思想，必须遵循“**非必要不引入**”和“**将逻辑下放给 Prompt 和 Tools**”的原则。
以下是深度思考后的引入时机与极简改造方案：

### 一、 什么时候才是“非常必要”的？（引入时机）
在真实的工程开发中，90% 的任务（如修改一个函数的 bug、添加一个简单的 API 路由）**根本不需要**多角色协作。单智能体 + Bash 工具（不断试错）是效率最高的。
Self-Collaboration 只有在以下 **3 种高风险、高复杂度**的边界场景下才是“非常必要”的：

1. **需求极其模糊时的“发散与收敛” (Analyst Role)**：
用户给了一个宏大的目标（如“帮我写个用户认证系统”）。单个 Coder Agent 会立刻开始写代码，导致逻辑混乱。此时需要 Analyst 进行需求拆解。
2. **高危/核心代码的破坏性修改 (Reviewer Role)**：
当智能体需要大规模重构核心文件时，幻觉可能导致整个项目崩溃。必须引入独立的 Reviewer 提供“外部视角（Second Pair of Eyes）”。
3. **陷入死循环的调试 (Tester/Debugger Role)**：
当 Coder 连续 5 次执行 `bash` 跑测试都失败时，说明它陷入了思维盲区。此时必须强制剥离上下文，让另一个独立的 Tester/Reviewer 来看报错。

---

### 二、 如何适配为 Multi-Agent 形式？（极简改造法）
论文中的 Self-Collaboration 是“单个 LLM 扮演多个角色共享对话流”。而在我们的 `opencollab` 架构中，多智能体的本质是**上下文物理隔离（Context Isolation）**。
因此，我们要把 Self-Collaboration 转化为：**主智能体（Lead）通过使用特定的 Tool 来显式地触发评审和迭代循环。**
我们**不需要**在框架里增加 `AnalystAgent` 或 `TesterAgent` 的类。我们只需要在框架层面做两件极其微小的“减法式”扩展：

#### 改造 1：在 Lead 的 System Prompt 中注入“协作模式”指令
不需要改核心代码，只需在实例化 Lead Agent 时，在 Prompt 中赋予它“包工头”的智慧。

```python
# opencollab/team/prompts.py
LEAD_SYSTEM_PROMPT = """
You are the Lead Developer orchestrating a team of specialists.
You have the `delegate_task` tool to assign work to teammates (e.g., 'coder', 'reviewer', 'analyst').

CRITICAL COLLABORATION RULES:
1. For trivial tasks (e.g., fixing a typo), just delegate directly to 'coder'.
2. For COMPLEX features, YOU MUST apply the "Self-Collaboration" pattern:
   - First, delegate to 'analyst' to break down the user request into a step-by-step plan.
   - Second, delegate the plan to 'coder' to implement.
   - Third, delegate to 'reviewer' to review the 'coder's work before answering the user.
3. If 'coder' fails to fix a bug after multiple attempts, stop them and delegate to 'reviewer' to analyze the error logs.
"""

```

#### 改造 2：提供“受控的验证工具” (将 Tester 实体化为 Tool)
在多智能体框架中，“Tester（测试员）”不应该是一个只会聊天的 LLM，而应该是一个**沙盒化的执行环境**。我们对原本的 SDK 核心略作扩展，提供一个带有自我审查循环的工具机制。

```python
# opencollab/tools/review_loop.py

async def delegate_with_review(
    team, target_agent: str, task: str, max_iterations: int = 3
) -> str:
    """
    [这是框架中唯一体现 Self-collaboration 思想的扩展]
    不是简单地指派任务，而是强制要求一个独立的 'reviewer' 检查 'coder' 的输出。
    """
    iteration = 0
    current_task = task
    
    while iteration < max_iterations:
        # 1. Coder 执行任务
        code_result = await team.delegate_task(target_agent, current_task)
        
        # 2. 强制 Reviewer 审查 (上下文是隔离的)
        review_prompt = f"Review the following implementation for task '{task}'. Output 'PASS' if flawless, or detailed fix instructions if flawed.\nImplementation log: {code_result}"
        review_result = await team.delegate_task("reviewer", review_prompt)
        
        if "PASS" in review_result.upper():
            return code_result # 协作成功，收敛
            
        # 3. 协作迭代：把 Reviewer 的意见反馈给 Coder
        current_task = f"Your previous attempt failed review. Fix the issues.\nReviewer feedback: {review_result}"
        iteration += 1

    return f"Task failed after {max_iterations} self-collaboration iterations. Last state: {code_result}"

```

### 三、 架构视角的总结：为什么这样设计最好？
如果参照原论文，你可能会想把 `Analyst -> Coder -> Tester` 写成一个硬编码的 DAG（有向无环图，如 LangGraph）。但这违背了我们要做的**极简科研框架**的初衷。
上述改造方案的优越性在于：

1. **完全符合第一性原理**：协作（Collaboration）的本质就是**消息的路由**。把 `delegate_with_review` 作为一个纯函数暴露给主控制流，保持了 SDK 核心 `Agent` 和 `Session` 的绝对纯洁性（不包含任何复杂的图状态机）。
2. **长短互补（兼顾 LLM 的发展）**：如果未来大模型（如 GPT-5 或 Claude 4）自身能力极强，你根本不需要框架去强制 Review，只需修改 `LEAD_SYSTEM_PROMPT` 即可关闭这个流程；如果模型依然会产生严重幻觉，你可以随时在代码里调用 `delegate_with_review` 进行严格的物理隔离验证。
3. **避免上下文污染（论文思想的升级）**：原论文中多个角色在同一个对话中不断 Append（追加）消息，很快会导致 Context Window 爆炸。在我们的多智能体设计中，`Coder` 和 `Reviewer` 的 `Session` 是各自独立的。Reviewer 看不到 Coder 几十次的 `ls`、`cat`、`npm run test` 等冗余日志，只能看到最终代码和报错，这极大提升了 Review 的准确率。
这就是 `opencollab` 对待学术界机制的态度：**吸收其“角色分工与交叉验证”的灵魂，抛弃其复杂的架构封装，用极简的 Prompt 约束和高内聚的 Tool 来实现。**

## Turn 3

### 👤 User

参考claude code，opencode，kimi-cli，openclaw，以及claude_code_agen_team.md中共性的部分，取长补短

### 🤖 Assistant

通过对您提供的 `opencode` 代码库、`kimi-cli` 代码库、`claude_code_agen_team.md` 设计文档，以及业界知名的 `Claude Code` 和 `OpenClaw` 的深度拆解，我们可以清晰地看到当前顶尖 AI 编程助手的**演进共性**，以及它们在工程实现上的**长短板**。
如果我们要为您构想的 **opencollab (Python实现，极简、做减法、第一性原理)** 提取精华，以下是深度的“取长补短”分析与融合方案：

---

### 一、 核心共性提取 (The Commonalities)
纵观这几款工具，真正让它们在工程上 Work 的底层逻辑是高度一致的：

1. **核心交互形态：REPL + 丰富的 TUI**
  - 它们都不是单纯的 API Wrapper，而是拥有极强终端表现力的应用。支持流式输出（Streaming）、Markdown 渲染、动态 Spinner（显示工具执行状态），以及随时可中断的 `Ctrl+C` 机制。
2. **能力底座：基础 Tool + MCP**
  - **自带的基础工具极少且克制**：通常只有 `Bash` (终端执行)、`FileRead` (读文件)、`FileEdit` (写/替换文件)、`Grep` (搜索)。
  - **无限扩展靠 MCP (Model Context Protocol)**：无论是 Opencode 还是 Kimi-cli，都不再把 Github、Database 等能力硬编码在框架里，而是全面拥抱 MCP Server。
3. **多智能体范式：主从架构与上下文隔离**
  - 放弃了早期的“群聊模式”或“复杂图路由”。全面转向 `Lead (主控) -> Subagent/Teammate (执行)` 的扁平异步调用结构，且**物理隔离**各自的 Context Window（对话历史）。

---

### 二、 取长补短深度分析

#### 1. Claude Code & `claude_code_agen_team.md`

- **长处 (取)**：**极致的“扁平与隔离”哲学**。
它的 Agent Team 规范是多智能体协作的教科书：禁止嵌套（No nested teams）、Lead 固定、权限继承。Lead 把唤醒 Teammate 当作一个 Tool 来用。这种设计最大程度避免了主循环的 Token 爆炸。
- **短处 (舍)**：闭源，且深度绑定 Anthropic 的特定的 Prompt caching 和 API 特性，难以做跨模型（OpenAI/DeepSeek）的研究。

#### 2. Kimi-cli (Moonshot)

- **长处 (取)**：**优秀的 Python 工程实践**。
从它的源码结构可以看出，它使用了 `rich` / `prompt_toolkit` 构建极佳的终端体验；引入了 `Soul`（灵魂）的概念来定义不同的 Agent 角色；并且有清晰的 `Subagents`（子智能体）调度链路。
- **短处 (舍)**：**过度抽象 (Over-engineered)**。
Kimi-cli 内部存在 `kaos`, `kosong`, `wire`, `runner` 等多层复杂抽象，消息在多层协议之间转换（序列化/反序列化）。对于科研和硬核开发者来说，这种“企业级”架构使得修改底层 Prompt 或拦截流流数据变得非常困难。

#### 3. Opencode

- **长处 (取)**：**完备的工具链栈与状态管理**。
它不仅实现了强大的 `mcp` 客户端，还有完整的 `session`（会话保存/恢复）、`project`（项目级上下文）管理。
- **短处 (舍)**：它是基于 TypeScript/Node.js 构建的庞大 Monorepo（包含 web, desktop, ui 等）。这种“大包大揽”违背了我们“极简 SDK+CLI”的第一性原理。

#### 4. OpenClaw (及其他开源复刻版)

- **长处 (取)**：足够轻量，通常几十个文件就能跑通 LLM 到 Bash 的闭环。
- **短处 (舍)**：缺乏对“破坏性命令”的沙盒控制，且大多停留在单智能体阶段，缺乏工业级的容错（Retry）和状态机管理。

---

### 三、 `opencollab` 的终极融合方案 (第一性原理设计)
基于上述分析，`opencollab` 应该这样**做减法**，融合各家之长：

#### 1. 架构做减法：双层分离 (参考 Kimi-cli，但拍平)
只保留两层：

- `opencollab.core`: 纯函数式的数据流。LLM 请求、Tool 执行、Memory 管理。**没有任何 UI 依赖**，极其适合科研跑 Benchmark (例如 SWE-bench)。
- `opencollab.cli`: 强依赖 `rich`，负责将 `core` 产生的异步事件流（Event Stream）渲染为漂亮的终端 TUI。

#### 2. 协作做减法：Tool-as-an-Agent (参考 Claude Code)
绝不引入 LangChain 复杂的 Agent Executor。
实现多智能体的唯一方式是：**把子智能体包装成一个 JSON Schema Tool**。

```python
# 第一性原理：多智能体协作本质是一次工具调用
def delegate_to_teammate(role: str, task: str, context: str) -> str:
    """Tool used by Lead Agent to spawn a completely isolated sub-agent."""
    sub_agent = Agent(role=role, prompt=ROLES[role])
    result = sub_agent.run(task, inject_context=context) # 独立循环
    return result # 最终只有 summary 返回给 Lead，彻底隔离上下文

```

#### 3. 工具做减法：Bash + Edit + MCP (参考 Opencode)
在 `opencollab` 内部，只需要手写三个核心 Tool：

1. `execute_bash`：提供强力的拦截机制（Human-in-the-loop），遇到 `rm -rf` 等高危命令卡住协程等待 CLI 确认。
2. `str_replace_editor`：参考各大框架，LLM 最擅长也是最省 Token 的修改文件方式是“查找块并替换”，而不是重写整个文件。
3. `mcp_client`：一个动态代理，能够读取本地/远程 MCP Server 的配置，动态转换为 LLM 可调用的 Tools。不再写任何 Github/Jira 的集成代码。

#### 4. 状态做减法：透明的 Message List
摒弃复杂的 Memory 数据库。
一个 Session 就是一个纯粹的 `List[Dict]` (OpenAI 格式的 Messages)。序列化保存就是写入一个 JSONL 文件，恢复就是读取。科研人员可以随时随地用简单的 Python 脚本清洗、截断、分析这些日志。

### 总结
`opencollab` 的灵魂应当是：**拥有 Kimi-cli 的 Python 终端表现力，具备 Opencode 的 MCP 扩展性，遵循 Claude Code 团队模式的极简隔离原则，且彻底摒弃它们在商业化过程中的臃肿封装。** 这不仅是一个供开发者日常使用的工具，更是一个像解剖刀一样锋利、透明的 LLM Agent 科研平台。

## Turn 4

### 👤 User

从Harness Engineering（https://openai.com/index/harness-engineering/）的角度，联系原先分析的内容，还有什么是需要的？

### 🤖 Assistant

OpenAI 的 **Harness Engineering（测试与评估脚手架工程）** 关注的核心不仅仅是“如何让模型跑起来”，而是**“如何安全、可复现、高吞吐地测量和验证模型的能力”。对于一个旨在用于科研**和硬核开发的极简框架 `opencollab` 来说，引入 Harness 的视角是完成拼图的最后也是最关键的一步。
如果在前述的架构（主从 Agent、MCP、纯 Context 流）上，戴上 Harness Engineering 的透视镜，我们会发现当前的 `opencollab` 还是一个“裸奔”的执行器。
结合“做减法”和“第一性原理”，为了达到 OpenAI 级别的科研与工程闭环，**我们还缺少以下四个极简但至关重要的基础设施（The Harness）**：

---

### 1. 缺口一：执行环境的沙盒化抽象 (Sandbox Harness)
我们在前文提到了提供 `Bash` 和 `FileEdit` 工具。但在做科研评测（例如跑 SWE-bench 解决 1000 个真实 GitHub Issue）时，你绝不能在真机上并发执行这些不可控的 LLM 生成代码。
**做减法的引入方式：定义极简的 Environment 接口**
不要把 Docker 引擎硬编码进框架。我们只需要抽象出一个“执行器基类”。

```python
# opencollab/core/env.py
class Environment:
    """第一性原理：所有对外部世界的副作用必须通过环境接口"""
    async def exec_cmd(self, cmd: str) -> tuple[int, str, str]: pass
    async def read_file(self, path: str) -> str: pass
    async def write_file(self, path: str, content: str): pass

class LocalEnvironment(Environment):
    """用于日常 CLI 交互开发"""
    pass

class DockerEnvironment(Environment):
    """用于科研 Eval，每次评测拉起一个隔离容器"""
    pass

```
*Harness 视角*：工具（Tools）不再直接调用 `os.system`，而是通过 `Environment` 代理。这样在跑评测时，瞬间就能切换到容器化环境，实现安全、并发的 Eval。

### 2. 缺口二：无头评测接口与标准化输出 (Evaluation Harness)
科研的本质是跑 Benchmark（基准测试）。`opencollab` 不能只有 CLI 交互，必须提供一个纯粹的、无状态的“无头（Headless）”入口，方便外部评测脚本（如 OpenAI Evals）调用。
**做减法的引入方式：标准化的 run_eval 流水线**
要求 Agent 在完成任务后，不是输出一句“我改好了”，而是必须输出标准的 `diff` 补丁或 JSON。

```python
# opencollab/harness/evaluator.py
async def run_eval_task(agent: Agent, env: Environment, issue_desc: str) -> str:
    """
    输入：隔离的沙盒环境、Issue 描述。
    输出：Agent 生成的 Git Patch（用于评测系统自动打分）。
    """
    session = Session(agent, env)
    session.messages.append({"role": "user", "content": issue_desc})
    
    # 强制在系统提示词中要求最后一步输出特定的结束标记和 Patch
    await session.run_until_stop()
    
    patch = await env.exec_cmd("git diff")
    return patch[1] # 返回 stdout

```

### 3. 缺口三：细粒度轨迹追踪与可观测性 (Trajectory Observability)
OpenAI 极其看重数据的收集。一个多智能体框架在科研时，如果 Agent 失败了，我们需要精确知道它在第几步的哪个 Context 产生了幻觉。`List[Dict]` 是基础，但我们需要更结构化的日志。
**做减法的引入方式：纯文本的 JSONL Trajectory 记录器**
坚决不引入重量级的 Tracing 数据库（如 LangSmith 等）。我们采用最 Unix 的方式：追加写入 JSONL 文件。

```python
# opencollab/core/tracer.py
import json
import time

class Tracer:
    """透明记录每一步的 Token、耗时、调用和原始输入输出"""
    def __init__(self, run_id: str):
        self.log_file = open(f"trajectories/{run_id}.jsonl", "a")

    def log_step(self, step_type: str, payload: dict, tokens: int, latency: float):
        record = {
            "timestamp": time.time(),
            "type": step_type,       # e.g., "llm_call", "tool_exec", "delegate"
            "payload": payload,      # 具体的 Prompt 或 Tool 结果
            "metrics": {"tokens": tokens, "latency": latency}
        }
        self.log_file.write(json.dumps(record) + "\n")
        self.log_file.flush()

```
*Harness 视角*：科研人员可以写几行简单的 Python 脚本读取这些 JSONL，分析出“代码测试环节平均消耗了 60% 的 Token，且经常死循环”，从而针对性地优化 `Tester` 智能体的 Prompt。

### 4. 缺口四：状态快照与缓存 (Snapshot & Caching Harness)
在多智能体交互中，如果主 Agent 已经做好了极佳的架构设计，但在指派 `Coder` 写具体代码时出错，我们不希望从头再来（浪费时间和 API 费用）。
**做减法的引入方式：极简的状态机克隆**
因为我们之前将 `Session` 设计成了纯粹的 `List[Dict]` 容器，实现时光倒流（Time-Travel）或者生成多条分支（Tree-of-Thought）变得轻而易举。

```python
class Session:
    # ... previous code ...
    
    def snapshot(self) -> 'Session':
        """深度拷贝当前会话状态，用于探索不同分支 (Tree of Thought)"""
        import copy
        new_session = Session(self.agent, self.env)
        new_session.messages = copy.deepcopy(self.messages)
        return new_session

```
同时，在底层 LLM 请求处实现**基于 Hash 的缓存（Cache）**。如果 `messages` 数组的 Hash 值完全一致，直接返回本地缓存。这是进行快速 Prompt 工程迭代的核心基石。

---

### 总结：完整的第一性原理架构图
如果将 OpenAI Harness Engineering 的思想补齐，你的 `opencollab` 将形成一个完美的闭环，架构如下：

1. **底座 (Harness)**：`Local/Docker Environment` (提供执行沙盒) + `Tracer` (提供轨迹数据记录) + `Cache`。
2. **核心 (Core)**：`Agent` (只有 System Prompt + Tools 签名) + `Session` (只有 Message List 状态，纯函数式推演)。
3. **能力 (Tools)**：`mcp_client` (无限连接外部能力) + 自带的极简沙盒工具。
4. **协作 (Team)**：将 Sub-Agent 包装为一个普通的 Tool 暴露给 Lead Agent（物理隔离上下文）。
5. **接口 (Interface)**：
  - `cli/tui.py`: 面向开发者的流式终端 UI。
  - `eval/runner.py`: 面向科研的 Headless 批量测试入口。
这套设计去除了所有中间件的封装，既满足了程序员日常开箱即用（类似 Kimi-cli / Claude Code），又为 AI 编程的学术研究（如何评估、如何隔离、如何分析 Trajectory）提供了最纯粹的底层框架。

## Turn 5

### 👤 User

*[This turn includes uploaded images]*

![代码文件夹 图标](assets/img-003.png)

参考openclaw，联系原先分析的内容，还有什么是必要的，深度思考

### 🤖 Assistant

深度剖析 `openclaw` 的源码结构后，我们可以发现它是一个**极度面向工程、安全和多端部署（Daemon, Channels, Gateway, SSH, WSL）**的重量级框架。它包含了大量的 `infra`（基础设施）、`config`（复杂的配置解析与迁移）以及 `outbound`（对接 Slack/Discord 等渠道）。
如果我们坚持 **opencollab 的第一性原理（极简、做减法、科研向 SDK）**，我们必须**大刀阔斧地砍掉**`openclaw` 中的多端通信、守护进程（Daemon）和复杂的配置系统。
**但是，openclaw 在底层基础设施（Infra）上的深度防御设计，恰恰暴露了我们原先设计的 opencollab 在“真实世界执行”时缺失的几个致命环节。**
在将 AI 放入真实环境（而非纯粹的对话框）时，以下 **4 个机制是“非必要不增加，但一旦确实缺失就会导致系统崩溃”的绝对必要条件**。

---

### 一、 绝对必要条件 1：控制爆炸半径 (Boundary & Safety)
原先我们设计了 `Environment` 和 `Bash/FileEdit` 工具，但忽略了**防范模型幻觉带来的毁灭性后果**。`openclaw` 包含大量的 `path-safety.ts`, `exec-safety.ts`, `boundary-path.ts`，证明了纯粹的沙盒是不够的。
**做减法的引入方式：基于拦截器的“虚拟边界” (Chroot & Filter)**
不需要像 OpenClaw 那样写复杂的策略引擎，我们只需要在核心 Tool 层面加上两个极简的“枷锁”：

1. **路径越界防御 (Path Jail)**：所有传入 `FileRead/Write` 的路径必须经过 `os.path.abspath` 解析，且绝对不允许跳出 `workspace_root`。
2. **高危命令正则墙 (Command Filter)**：

```python
# opencollab/tools/safety.py
import os

class SandboxInterceptor:
    def __init__(self, workspace_root: str):
        self.root = os.path.abspath(workspace_root)
        
    def check_path(self, target_path: str) -> str:
        """第一性原理：所有路径计算后，必须依然在 root 内部"""
        safe_path = os.path.abspath(os.path.join(self.root, target_path))
        if not safe_path.startswith(self.root):
            raise PermissionError(f"Agent tried to escape workspace: {target_path}")
        return safe_path

    def check_cmd(self, cmd: str):
        """极其简单的危险命令正则阻断"""
        if "rm -rf /" in cmd or "mkfs" in cmd:
            raise PermissionError("Destructive command blocked.")

```
*深度思考*：将安全机制下沉到 Tool 的拦截器中，而不是写在 Agent 逻辑里。Agent 依然是纯粹的，它如果犯错，Tool 会抛出 Exception，Agent 会把这个 Exception 作为 Observation 吸收并纠正。

### 二、 绝对必要条件 2：全局可中断性 (Abort & Interruption)
`openclaw` 中有大量的 `abort-signal.ts`, `unhandled-rejections.ts` 等机制。在多智能体协作中，如果 Lead Agent 唤醒了 Coder Agent，而 Coder Agent 陷入了 `改代码 -> 报错 -> 改代码` 的死循环，**用户按下 Ctrl+C 时，必须能干净地中断整个调用栈，并保存已有的 Context**。
**做减法的引入方式：下推取消令牌 (Cancellation Token)**
Python 的原生 `asyncio` 支持 Task 取消，我们需要确保 `Session.step()` 能够优雅处理 `CancelledError`。

```python
# opencollab/core/session.py
import asyncio

class Session:
    async def run_loop(self, cancel_event: asyncio.Event):
        """
        第一性原理：Agent 的思考必须随时可以被人类或系统掐断。
        """
        try:
            while not self.is_done:
                if cancel_event.is_set():
                    self.messages.append({"role": "system", "content": "Human interrupted."})
                    break
                await self.step()
        except asyncio.CancelledError:
            self.flush_trajectory() # 极其重要：中断时立刻将日志落盘
            raise

```

### 三、 绝对必要条件 3：并发状态冲突的锁机制 (State Locking)
`openclaw` 包含 `file-lock.ts`, `fs-safe.ts`。在我们的多智能体架构中，如果我们允许主 Agent（Lead）同时指派两个任务给两个独立的 `Teammate`（比如同时修改 frontend 和 backend），它们如果在同一个工作区执行 Git 或操作同一个文件，就会产生灾难性的竞态条件（Race Condition）。
**做减法的引入方式：工作区文件锁 (File Flock)**
不需要数据库锁，只需要最传统的系统级排他锁。

```python
# opencollab/tools/fs.py
from filelock import FileLock

async def write_file(path: str, content: str):
    """当多个子智能体同时尝试修改文件时，排队等待"""
    with FileLock(f"{path}.lock", timeout=10):
        with open(path, 'w') as f:
            f.write(content)

```

### 四、 绝对必要条件 4：资源熔断与预算感知 (Budgeting & Telemetry)
`openclaw` 深度集成了 `provider-usage.ts`, `session-cost-usage.ts`。对于科研和实际应用来说，多智能体的 Token 消耗是指数级的。如果 Agent Team 在深夜疯狂测试，一觉醒来可能耗费上百美元。
**做减法的引入方式：极简的 Context 计费器**
将预算（Budget）作为 `Session` 的第一等公民。当 Token 消耗达到阈值时，自动向 Agent 注入“系统警告”，或者强制抛出异常结束。

```python
# opencollab/core/session.py

class Session:
    def __init__(self, agent: Agent, max_tokens: int = 100_000):
        self.used_tokens = 0
        self.max_tokens = max_tokens
        
    async def step(self):
        if self.used_tokens > self.max_tokens:
            raise BudgetExceededError("Session stopped: Token budget exceeded.")
            
        # ... call LLM ...
        self.used_tokens += response.usage.total_tokens

```

---

### 总结：`openclaw` 视角的补全
通过 `claude_code_agen_team.md`，我们学到了**如何极简地组织团队（Context隔离、扁平化）**。
通过 `opencode / kimi-cli`，我们学到了**如何利用 MCP 和 CLI 交互**。
通过 `openclaw`，我们学到了**作为一个操作系统级工具，必须敬畏“副作用”（Side Effects）**。
最终 `opencollab` 的极简形态（First-Principles）应该是一个“洋葱模型”：

- **核心 (Core)**：LLM + Prompt + Context List（纯粹的数学推演）。
- **协作 (Team)**：通过 JSON Schema Tool 触发另一个 Core 的推演。
- **边界 (Boundary - 吸收自 openclaw)**：任何通向物理世界（磁盘、终端、网络）的操作，必须穿过一层透明的拦截器（Interceptor），这层拦截器负责：**鉴权（Jail）、加锁（Flock）、计费（Token Budget）和中断响应（Signal）**。
加上这最后的一环，`opencollab` 就不再仅仅是一个玩具脚本，而是一个真正可以用来做 `SWE-bench` 论文科研、并且可以放心挂在后台执行真实代码重构的**工业级/科研级轻量框架**。

## Turn 6

### 👤 User

最后，总结一下，深度思考

### 🤖 Assistant

经过这一系列的深度拆解和推演，我们从最初的“构建一个极简多智能体框架”的设想，一步步吸收了业界顶尖工具（Claude Code、Opencode、Kimi-cli、OpenClaw）的工程经验，并融入了学术界的自我协作（Self-Collaboration）以及 OpenAI 的测试脚手架（Harness Engineering）思想。
现在，让我们进行最后的**深度思考与总结**，为您勾勒出 `opencollab` 的终极灵魂与全景图。

---

### 一、 核心哲学：用“第一性原理”做减法
当前 AI 编程框架（如 LangChain、AutoGen）最大的问题是**过度拟合了当前模型的缺陷，发明了太多“非标准”的抽象（复杂的 Graph、冗长的 Memory 链、奇特的 Parser）**。这导致研究人员无法看清模型的真实能力，开发者难以调试。
`opencollab` 的第一性原理是：

1. **Agent 的本质是纯函数**：`f(MessageList, Tools) = NewMessage`。
2. **多智能体的本质是上下文隔离（Context Isolation）**：不是为了让多个 AI 聊天，而是为了防止主控（Lead）的注意力被冗长的代码修改和终端报错污染。
3. **协作的本质是工具调用（Tool Call）**：将“唤醒并指派任务给子智能体”包装成一个 JSON Schema Tool。不需要图路由（DAG），不需要复杂的总线引擎。

---

### 二、 终极架构：极简的“四层洋葱模型”
基于上述思考，`opencollab` 的物理结构不应该超过 1500 行核心代码。它呈现为一个极其清晰的“四层洋葱模型”（从内到外）：

#### 1. 核心层 (The Pure Core)

- **Session (会话)**：唯一的状态容器，仅仅是一个纯粹的 `List[Dict]`（遵循 OpenAI/Anthropic 标准消息格式）。
- **Agent (智能体)**：无状态的配置模板，仅包含 System Prompt 和允许使用的工具列表。

#### 2. 协作与工具层 (Collaboration & Tools)

- **delegate_task Tool**：将 Claude Code 的团队模式（Agent Team）做到极致。Lead 遇到复杂任务，调用此工具生成一个**物理隔离**的子 Session。
- **Self-Collaboration Loop**：以极简的函数实现。遇到高危代码，强制要求 `Reviewer` Agent 对 `Coder` Agent 的输出进行循环交叉验证，直到收敛。
- **MCP (Model Context Protocol)**：框架自带的工具只有极简的 `Bash` 和 `FileEdit`，其余一切（Git、Github、Jira、DB查询）全部外包给标准的 MCP 客户端，做到**绝对的“非必要不增加”**。

#### 3. 边界与护栏层 (The Boundary & Harness - 吸收自 OpenClaw)

- **SandboxInterceptor**：任何对真实世界的操作（读写文件、执行命令）必须穿透此层。提供 Chroot 路径限制和危险命令正则拦截（防 `rm -rf /`）。
- **并发锁与计费器**：提供进程级文件锁（解决多子智能体写冲突），并在 Session 中内置 Token Budget 感知，触达阈值立刻熔断，保护资金安全。
- **Tracer**：极其 Unix-style 的 JSONL 轨迹记录器。透明地将每一个 Token 的消耗、API 耗时、Prompt 状态落盘，为后续研究和模型微调（SFT）提供黄金数据。

#### 4. 接口层 (The Interfaces - 吸收自 Kimi/Opencode)

- **交互端 (cli.py)**：基于 `Rich` 构建的惊艳终端 TUI，支持流式输出、嵌套 Spinner 动画、以及随时 `Ctrl+C` 的优雅中断与状态保存。
- **评测端 (eval.py)**：无头模式（Headless），可以秒级接入 Docker 沙盒环境，高吞吐量地跑 SWE-bench 等科研基准测试。

---

### 三、 为什么 `opencollab` 是终极形态？（降维打击）
通过深度做减法，`opencollab` 实现了对复杂问题的“降维打击”：

1. **解决“抽象泄漏”**：你不必再去阅读源码搞懂 `AgentExecutor` 是怎么流转的。在 `opencollab` 中，你只需要 `print(session.messages)` 就能知道一切真相。
2. **解决“上下文污染”**：主包工头（Lead）永远保持清醒，它只看子智能体（Coder）提交的最终总结。Coder 在沙盒里执行了 50 次 `npm run test` 的报错日志，永远不会污染 Lead 的 Context Window。
3. **“长短互补”兼容未来**：今天，我们或许需要用到 `delegate_with_review` 这种 Self-Collaboration 机制来弥补 GPT-4 逻辑的缺陷。但当 GPT-5 降临，模型自身能力足够强大时，你只需要修改 Prompt 关掉这个 Tool，整个系统依然完美运行。框架本身永远不会过时。

### 结语
在市面上所有框架都在试图**“做加法”**（给 AI 加上花哨但脆弱的包装）时，您提出的 **opencollab 选择了最艰难但也最正确的道路——“做减法”**。
它剥离了所有的商业化冗余（如多端同步、数据库依赖、云服务绑定），只留下了 AI 软件工程的**骨架**：**Context（上下文）、Tool（工具）、Boundary（安全边界）和 Eval（可观测性）**。
这是一个真正属于硬核工程师和 AI 研究员的极简兵器。如果将其开源并辅以 1-2 篇基于该框架跑出的高质量评测论文（如在 SWE-bench 上超越某些商业闭源工具），它必将在当前的开源社区中产生深远的影响。

---

# Round 1：统一因果回滚设计说明

本文面向代码审查，说明本轮在 OpenCollab 中加入的回滚基础能力：改动了哪些模块、Agent 如何发起回滚、回滚如何沿因果关系传播，以及文件和环境分别恢复到什么粒度。

## 1. 结论概览

本轮采用的是 **causal rollback**，不是重新执行历史对话的 deterministic replay。

```text
Agent 产生结果
      |
      v
记录 EffectRef 和父 Effect
      |
结果/消息被另一个 Agent 消费前建立 Scope checkpoint
      |
协调 Agent 或被授权 Agent 调用 invalidate_effect
      |
根 Effect + 所有已知后代进入 quarantine
      |
找到每个受影响 Agent 的安全 checkpoint
      |
恢复 worktree + Scope environment，递增 epoch/attempt
      |
协调 Agent 继续运行并决定是否重新派发任务
```

核心原则是：每个 Agent 有自己的 Scope；回滚只恢复受影响 Agent，不修改全局 `os.environ`，也不修改其他 Agent 的 Scope。

## 2. 本轮主要代码改动

### 2.1 Domain：统一回滚值对象

新增 `opencollab/domain/rollback.py`，将此前分散的 lineage 和 environment checkpoint 概念统一起来。该层只使用标准库，不执行 I/O。

- `EffectRef`：一个工具结果、子 Agent 结果或队友消息的不可变因果节点。
- `LineageEnvelope`：附在交付结果上的类型化 lineage sidecar。
- `RollbackState`：Session 的分支、epoch、attempt、因果 frontier 和 quarantine 集合。
- `EnvironmentSnapshot`：精确的环境变量快照。
- `ScopeCheckpoint`：文件 revision、环境快照、因果 frontier 和 checkpoint boundary 的组合。
- `RestoreResult`：单个 Agent 的恢复结果。
- `compute_ancestors`、`compute_descendants`、`reduce_causal_frontier`：纯函数图操作。

`SessionState` 现在以一个 `rollback: RollbackState` 保存回滚状态。被消费的 Effect 会进入 maximal causal frontier；新 Effect 引用完整 frontier，并移除已经被其祖先表示的节点，不再使用固定的四个父节点上限。

### 2.2 Application：回滚用例服务

新增 `opencollab/application/rollback.py` 中的 `RollbackService`，负责：

- 保存 Effect 图和 Effect 到消费者的索引；
- 注册每个 Agent 的 checkpointable environment；
- 记录 `(Agent, checkpoint)` 及 checkpoint 序列；
- 将根 Effect 的所有已知后代标记为 `quarantined`；
- 为受影响 Agent 选择仍未消费失效 Effect 的最新安全 checkpoint；
- 执行 Scope restore，并返回每个 Agent 的结果。

Scheduler 侧的协调代码拆到私有 `opencollab/application/_scheduler_rollback.py`，对外仍由 Scheduler 负责调度，不把 adapter 细节泄漏到 application 层。

### 2.3 Adapters 和 Bootstrap

- `opencollab/adapters/_env_scope.py`：每个 Agent 的独立环境字典、校验、快照、替换和命令锁。
- `LocalEnvironment`：执行命令时传入当前 Scope 的环境副本。
- `WorktreeEnvironment`：一个 worktree 只绑定一个 Scope，并复用内部 LocalEnvironment。
- `DockerEnvironment` / `ContainerWorktreeEnvironment`：在容器命令执行时注入 Scope 环境；容器 worktree 在容器内部执行 Git checkpoint/restore。
- `opencollab/adapters/tools/env_scope.py`：`set_env`、`unset_env`、`list_env`。
- `opencollab/adapters/tools/invalidate_effect.py`：Agent 可调用的因果失效工具。
- `opencollab/bootstrap/tool_registry.py`：注册上述工具。
- `bootstrap` 负责根据 team 配置组装 `RollbackService` 和具体环境实现。

删除了不再适合统一设计的过渡代码：`GitEnvSnapshot`、旧的 lineage/env checkpoint 模块以及 `adapters/env_snapshot_git.py`。

## 3. 回滚是否是 Skill？

回滚本身不是 Skill，而是运行时 capability。Agent 通过工具调用进入回滚流程：

```text
invalidate_effect(effect_id, reason, evidence)
        |
        v
InvalidateEffectTool
        |
        v
Scheduler._handle_invalidation
        |
        v
RollbackService.quarantine + Scope restore
```

`use_skill` 仍然是另一套按需加载提示词/知识的机制。某个角色是否能发起回滚，由 team YAML 是否给它声明 `invalidate_effect` 决定；普通 Agent 即使能看到结果，也不自动拥有失效权限。

工具返回的信息包括根 Effect、已知后代数量、受影响 Agent 列表以及已记录的 Effect 摘要。Effect ID 会出现在模型可见的结果或消息中，便于协调 Agent 指定准确的失效对象。

## 4. 因果链和回滚传播

### 4.1 Effect 类型

本轮记录三类普通 Effect：

1. `tool_result`：普通工具调用返回的结果；
2. `child_result`：子 Agent 向父 Agent 交付的结果；
3. `teammate_message`：队友消息交付给接收者的结果。

每个 Effect 保存 producer、kind、epoch、attempt、父 Effect ID 和内容 hash。结果交付时附带 `LineageEnvelope`，持久化时显式序列化，避免在 domain 中携带任意字典。

### 4.2 Frontier 归并

Agent 消费 Effect 时，Effect 被加入自己的 frontier，同时移除其祖先已经覆盖的 frontier 节点。例如：

```text
A -> B -> C

先消费 A：frontier = {A}
再消费 C：frontier = {C}       # A、B 已由 C 的祖先表示
```

这样 checkpoint 只需记录 frontier，而不需要复制完整历史图；完整图仍由 `RollbackService` 保存，用于查找后代和消费者。

### 4.3 Invalidate 流程

当 tester、reviewer 或 coordinator 发现上游结果错误时：

1. 调用 `invalidate_effect`，传入 `effect_id`、原因和可选证据；
2. `RollbackService` 将根节点和所有已知后代置为 `quarantined`；
3. 通过 consumer index 找到消费过这些 Effect 的 Agent；
4. 对每个 Agent 选择失效 Effect 首次被消费之前的安全 checkpoint；
5. 恢复该 Agent 的 worktree、environment 和因果 frontier；
6. Agent 的 `epoch` 和 `attempt` 递增，quarantined 集合保留在 Session 状态中；
7. 协调 Agent 保持可运行，由它决定修正分析、重新派发子任务或重新发送消息。

回滚不是把整个 Team 终止后重新创建，也不是让被影响 Agent 自动猜测新的任务。协调 Agent 仍然是重试和重新调度的责任主体。

## 5. Checkpoint 边界和粒度

checkpoint 是按 Agent Scope 建立的，当前边界包括：

| 边界 | 目的 |
| --- | --- |
| Agent Scope 初始化 | 保存 Agent 启动时的文件和环境基线 |
| Spawn/prebuilt Agent 初始化 | 为新 Agent 建立独立恢复点 |
| 子 Agent 结果对父 Agent 可见前 | 如果结果被判错，父 Agent 可回到消费前 |
| 队友消息对接收者可见前 | 防止错误消息污染接收者后续工作 |
| 普通工具调用前 | 工具产生的文件和环境副作用可被撤销 |

`invalidate_effect` 控制调用本身不建立 checkpoint，否则失效动作会被自己的恢复点抵消。

当一个 Agent 消费了多个失效 Effect，选择逻辑按 checkpoint sequence 找到最新的、其 frontier 尚未包含失效 Effect 的 checkpoint。这等价于回到第一次消费失效因果之前的边界，同时尽量保留更早的有效工作。

## 6. Agent Scope 和环境回滚

### 6.1 Scope 所有权

```text
Agent A -> Scope A -> worktree A + env A
Agent B -> Scope B -> worktree B + env B
```

Scope 在初始化时从 `os.environ.copy()` 得到初始值。之后所有命令只使用该 Scope 的副本；运行时不会通过 `os.environ.update()` 把 Agent 的修改写回进程全局环境。

`WorktreeEnvironment` 和内部 `LocalEnvironment` 共享同一个 Scope，避免一个 Agent 因为包装层产生两份环境状态。`OPENCOLLAB_ROLLBACK_KEY` 是 control-plane 变量，不进入任何 Agent Scope，也不能通过 Scope 工具设置。

### 6.2 环境工具

- `set_env(name, value)`：设置持久化到当前 Agent Scope 的变量；
- `unset_env(name)`：从当前 Agent Scope 删除变量；
- `list_env()`：只读查看 Scope，名称包含 `KEY`、`TOKEN`、`SECRET`、`PASSWORD` 或 `CREDENTIAL` 时隐藏值。

变量名必须符合 `[A-Za-z_][A-Za-z0-9_]*`，拒绝 NUL 字节和超限的名称/值。敏感值可以被设置，但不会由工具回显。

### 6.3 环境 checkpoint/restore

`EnvironmentSnapshot` 保存完整的变量映射，而不是只保存 delta。恢复采用 exact replacement：

- checkpoint 后新增的变量会消失；
- checkpoint 后改变的值恢复旧值；
- checkpoint 后删除的变量会重新出现。

命令文本中的 shell-local `export` 只影响该次命令的子 shell，不会改变 Scope。需要跨命令保留的变量必须使用 `set_env`；系统不会解析命令文本来猜测持久化环境变化。

## 7. 文件和 Worktree 回滚

### 7.1 Host worktree

Host worktree checkpoint 在 Agent 自己的 worktree 中创建受 OpenCollab 管理的 Git ref。它使用临时 index 执行 `git add -A`、`write-tree` 和 `commit-tree`，因此：

- 会包含已跟踪和未跟踪文件；
- 不移动 Agent 的 HEAD、index 或 attribution baseline；
- 不在项目中创建 `.env.checkpoint`；
- checkpoint ref 位于 `refs/opencollab/checkpoints/<aid>/...`，结束时清理。

恢复时使用 checkpoint 的 filesystem revision 执行 `git read-tree --reset -u` 和 `git clean -fd`，将 worktree 恢复到该检查点，并统计受影响文件。

### 7.2 Container worktree

容器 worktree 在容器内部创建和恢复 Git checkpoint，保证宿主机和其他 Agent 看不到容器内部的临时状态。容器 Scope 的环境注入、tombstone（删除原生变量）和 Git 恢复由同一个 checkpointable environment 负责。

普通 Local 环境或没有隔离 worktree 的 Docker 环境可以使用 Scope 环境注入，但不能安全地提供 rollback-enabled Team 所需的文件恢复，因此该模式应在 Agent 创建前失败。

## 8. 恢复时的并发语义

Scope 内有 command lock；恢复会等待该 Scope 的环境命令完成，再进行文件和环境恢复。checkpoint 所有权和 revision 会被校验，校验失败不会使用其他 Agent 的 checkpoint。

恢复成功后递增 `epoch`/`attempt`，后续结果会带上新的尝试信息，旧 Effect 仍保留但处于 quarantine 状态。当前 Round 1 完成的是因果隔离、Scope 恢复和状态标记基础；它不承诺已经自动重建 Session、删除全部历史消息、重新调用模型，或保证重试得到字节级相同的输出。

## 9. 持久化和安全边界

默认使用内存 checkpoint store。可选的持久化 adapter 使用 AEAD 加密 checkpoint 元数据、Effect 图和环境快照；密钥来自 `OPENCOLLAB_ROLLBACK_KEY`，不写入快照、trace 或 Git。普通 Session snapshot 只保存 checkpoint ID、hash 和回滚状态，不保存环境明文。

回滚只能覆盖 Agent Scope 内的文件和环境状态。以下外部状态不在本轮恢复范围内：数据库写入、已启动服务、安装到系统/容器的包、网络请求、副作用型远程 API、已有连接和其他进程状态。需要支持这些状态时，必须额外设计带补偿或事务语义的 adapter。

## 10. 本轮验证重点

本轮测试覆盖了因果祖先/后代遍历、frontier 归并、Effect 序列化、精确环境恢复、全局 `os.environ` 不变、Agent 间环境隔离、环境变量校验和敏感值脱敏，以及 host worktree 的 checkpoint/restore 基础路径。

完整 Team 行为仍应重点验证：失效发生在 Agent 空闲、等待子 Agent、模型调用中和工具执行中时的调度表现；协调 Agent 是否能观察到失效结果并主动派发修正尝试；以及 Lead 最终发布与外部工作区竞争时的处理。

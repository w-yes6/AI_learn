# LangChain 核心组件与工具调用

> 本文介绍 LangChain 的核心组件、工具调用、结构化输出和 Agent 执行流程，并以电价分析 Agent 为例说明完整的应用链路。

## 目录与学习目标

今天的学习分成 6 个部分：
1. LangChain 的基本组件：Model、Prompt、Tools。
2. Function Calling：模型如何决定调用工具。
3. 结构化输出：如何让模型返回程序可以处理的数据。
4. Agent 执行循环：如何完成多步任务。
5. 工具参数校验、超时、异常处理。
6. 实战：开发一个可以调用多个 Python 工具的 Agent。

## 一、先理解 LangChain 的整体工作方式
LangChain 是一个帮助开发者构建大模型应用和 Agent 的 Python 框架。它把模型调用、提示词、工具和结构化输出等能力组织成可复用的组件。

一个典型的 Agent 可以理解为下面这个流程：
用户提出任务

“分析昨天实时电价预测的误差”

Prompt + Model

理解任务，判断下一步需要做什么

Tools

查询数据、计算指标、分析异常时段

Agent 执行循环

观察工具结果，判断是否还需要调用工具

最终输出

返回计算结果和分析报告


这里需要区分两个角色：
- 大模型负责决策：理解问题、选择工具、生成参数。
- Python 程序负责执行：实际查询数据、计算指标、验证参数、处理错误。
LangChain 帮助你连接这些环节，但不意味着只要调用了 LangChain，程序就自动具备了可靠性。
## 二、Model：LangChain 如何调用大模型？
首先学习模型组件。

LangChain 不负责训练大模型，它负责提供统一的调用接口，让你可以用相对一致的方式调用不同厂商的模型。模型可以独立完成文本生成，也可以参与 Agent 的工具选择和任务执行。

> 资料来源：LangChain 官方文档。



### 2.1 基本代码
安装依赖：
```bash
pip install -U langchain langchain-openai python-dotenv pydantic
```


创建 `.env` 文件：
```dotenv
OPENAI_API_KEY=你的API密钥
OPENAI_MODEL=你的模型名称
```


然后调用模型：
```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model=os.environ["OPENAI_MODEL"],
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0,
    timeout=30,
    max_retries=2,
)

response = model.invoke("什么是日前电价？")

print(response.content)
```



这里有几个需要理解的参数。
| 参数 | 作用 |
| --- | --- |
| `model` | 指定使用哪个模型 |
| `api_key` | API 身份认证密钥 |
| `temperature` | 控制生成的随机性，具体效果取决于模型 |
| `timeout` | 请求超时时间，单位通常是秒 |
| `max_retries` | 请求失败后的最大重试次数 |

如果你使用的是其他厂商的模型，需要选择相应的 LangChain 集成。如果对方提供兼容 OpenAI 的接口，也可以配置 base_url，但具体兼容性需要确认。
### 2.2 `invoke()` 是什么？
你可以暂时把它理解为：向模型发送一次请求，并等待返回结果。
```python
response = model.invoke("计算 100 + 200")
```



这个操作只调用一次模型，并不会自动执行复杂的多步骤任务。

例如，模型可以告诉你 MAE 的计算公式，但它不会因为你问了一句“计算昨天的 MAE”，就自动读取你的本地 CSV。要完成后者，你需要为它提供工具。

这就引出了今天最重要的知识：Tools。
## 三、Prompt：如何让模型按要求工作？
在 LangChain 中，你可以通过系统提示词 system_prompt 指定 Agent 的角色、规则和行为边界。

例如：
```python
system_prompt = """
你是一名电价预测分析助手。

规则：
1. 计算预测误差时必须使用工具。
2. 不得编造预测值和实际值。
3. 数据不存在时，明确说明数据缺失。
4. 使用中文输出分析报告。
"""
```



在创建 Agent 时传入：
```python
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
)
```



这里的 tools 就是稍后定义的工具列表。

需要注意，Prompt 中写了规则，并不意味着这些规则一定会被遵守。比如“不能访问其他区域的数据”，还必须在工具函数内部进行权限检查。

Prompt 负责引导模型，程序负责执行强制约束。
## 四、Tools：如何让大模型调用 Python 函数？
这是今天最重要的部分。
### 4.1 什么是 Tool？
Tool 就是一个被封装成可调用接口的函数。

例如，你已经有一个计算电价预测误差的 Python 函数：
```python
def calculate_mae(actual, predicted):
    errors = [
        abs(a - p)
        for a, p in zip(actual, predicted)
    ]
    return sum(errors) / len(errors)
```



普通 Python 程序可以直接调用它：
```python
mae = calculate_mae(
    actual=[310, 300, 500],
    predicted=[300, 320, 450],
)
```



但大模型并不知道你的程序里有这个函数，也不能仅凭函数存在就直接执行它。

你需要通过 LangChain 把函数注册为工具，让模型知道：
- 工具叫什么名字。
- 工具可以解决什么问题。
- 工具需要什么参数。
- 工具返回什么结果。
### 4.2 使用 `@tool` 装饰器
LangChain 可以通过 @tool 将 Python 函数转换成工具。函数的类型注解用于定义输入参数的类型，文档字符串则帮助模型理解工具的用途。

> 资料来源：LangChain 官方文档。




例如：
```python
from langchain.tools import tool

@tool
def add_numbers(a: float, b: float) -> float:
    """计算两个数字的和。

    Args:
        a: 第一个数字。
        b: 第二个数字。
    """
    return a + b
```



这里有三个重点：
- @tool：将函数注册为 LangChain 工具。
- a: float, b: float：定义参数类型。
- 文档字符串：告诉模型什么时候使用工具，以及如何提供参数。
注册之后，模型就能根据工具的名称、描述和参数定义来决定是否调用它。
### 4.3 模型究竟如何调用工具？
假设用户说：
请计算 100 加 200。

模型可能返回一个工具调用请求：
```json
{
  "name": "add_numbers",
  "args": {
    "a": 100,
    "b": 200
  }
}
```


注意：这只是模型提出的调用请求，并不代表 Python 函数已经执行。

真正的过程是：
1. 用户提出问题

“计算 100 加 200”

2. 模型生成工具调用请求

工具名称：add_numbers

参数：a=100, b=200

3. LangChain 执行工具

Python 函数计算结果：300

4. 结果返回模型

模型整理并向用户回答：结果是 300。


这就是 Function Calling 的核心原理。模型提出工具调用，程序负责执行，工具结果再返回模型。

> 资料来源：LangChain 官方文档。
+1



### 4.4 工具应该怎么设计？
对于工业级 Agent，我建议你遵循四个原则。

（1）一个工具只负责一类明确的任务。

例如：
- get_price_data：查询电价数据。
- calculate_error_metrics：计算预测指标。
- find_worst_hours：查找误差最大的时段。
（2）工具的描述必须清晰。

不要只写“查询数据”，应该写明查询什么数据、需要什么参数、什么时候使用。

（3）工具的输入和输出要明确。

输入参数使用类型注解，复杂参数使用 Pydantic 校验，输出尽量采用结构化的数据。

（4）工具必须有权限边界。

不要让模型随意执行任意 SQL、Python 代码或文件操作。工具只开放完成任务所必需的能力。
## 五、结构化输出：让 Agent 的最终结果能够被程序处理
前面讲过 Pydantic。今天我们把它与 LangChain 结合起来。

假设你希望电价分析 Agent 最终输出以下字段：
- 分析日期。
- 市场类型。
- MAE。
- RMSE。
- 误差最大的时段。
- 分析结论。
先定义数据结构：
```python
from pydantic import BaseModel, Field
from typing import Literal

class PriceAnalysisReport(BaseModel):
    status: Literal["success", "insufficient_data"]
    date: str
    market: Literal["DA", "RT"]
    mae: float | None = None
    rmse: float | None = None
    worst_hours: list[str] = Field(
        default_factory=list
    )
    conclusion: str
```



这段代码定义了最终报告应该具备哪些字段，以及它们的类型。

在支持结构化输出的模型上，LangChain Agent 可以这样配置：
```python
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    response_format=PriceAnalysisReport,
)
```



调用后，可以读取结构化结果：
```python
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "分析 2026-10-26 的实时电价预测效果"
        }
    ]
})

report = result["structured_response"]
print(report.model_dump_json(indent=2))
```



response_format 可以使用 Pydantic 模型定义结果结构。LangChain 会根据模型能力选择适用的结构化输出策略，并将结果放在 structured_response 中。

> 资料来源：LangChain 官方文档。




注意：模型需要支持 Agent 工具调用与所选结构化输出策略的组合。如果使用的模型不支持同时进行这两种操作，可以先执行工具调用，再单独调用模型生成结构化报告。

另外，结构化输出只保证结果符合相应结构约束，不代表计算出来的 MAE 一定正确。MAE 和 RMSE 应该由 Python 根据真实数据计算，再由模型负责解释。
## 六、Agent 执行循环：为什么它能调用多个工具？
普通模型调用通常是一次请求、一次响应。

Agent 则会反复进行模型调用与工具执行，直到任务完成或达到执行限制。

一个典型的循环是：
1. 模型接收用户任务。
2. 模型判断下一步需要什么工具。
3. 程序执行工具。
4. 工具结果返回模型。
5. 模型决定继续调用工具，还是给出最终答案。
在 LangChain 当前版本中，可以使用 create_agent 创建 Agent。它会管理模型与工具之间的循环；底层使用 LangGraph 的执行机制。

Docs by LangChain




例如，用户说：
分析 2026 年 10 月 26 日的实时电价预测效果，计算 MAE 和 RMSE，并找出误差最大的两个时段。

Agent 可能执行以下步骤：
- 调用 get_price_data 获取预测值与实际值。
- 调用 calculate_error_metrics 计算 MAE 和 RMSE。
- 调用 find_worst_hours 找出误差最大的两个时段。
- 汇总结果，生成中文分析报告。
具体调用顺序由模型和你的流程约束决定，不应假设模型每次都会选择相同顺序。

重要：多个工具不一定意味着多个 Agent。 一个 Agent 就可以调用多个工具完成任务。
## 七、工具参数校验、超时和错误处理
这一部分决定了你的程序能否从演示代码走向可靠的业务系统。
### 7.1 参数校验
例如，工具只允许查询日前市场 DA 或实时市场 RT：
```python
from typing import Literal
from langchain.tools import tool

@tool
def get_price_data(
    date: str,
    market: Literal["DA", "RT"] = "RT",
) -> str:
    """查询指定日期的日前或实时电价数据。

    Args:
        date: 日期，格式为 YYYY-MM-DD。
        market: 市场类型，DA 表示日前，RT 表示实时。
    """
    # 这里还需要检查日期格式、
    # 数据权限以及该日期是否存在数据。
    return "这里返回查询结果"
```



Literal 可以把可选市场类型限制在指定值内，并帮助生成工具的参数 Schema。

不过，类型校验不能代替业务校验。你仍然需要检查日期是否合法、用户是否有权限访问对应区域、数据是否真实存在。
### 7.2 超时与重试
外部 API 或数据库可能响应缓慢，工具需要设置超时。

如果工具调用失败，可以针对可恢复的错误进行有限重试，但不要无条件重复执行所有操作。

尤其是发送邮件、下单、写入数据库等有副作用的工具，需要考虑幂等性，避免重试导致重复操作。
### 7.3 异常处理
例如，查询日期没有数据时，可以返回明确的错误：
```python
try:
    # 执行数据查询
    result = query_price_data(date)
except TimeoutError:
    result = {
        "error": "查询超时，请稍后重试"
    }
```
except ValueError as exc:
    result = {
        "error": str(exc)
    }



这只是异常处理的示意代码。真实项目中还要根据数据库驱动和外部服务的异常类型进行适配。

需要区分：
- 可恢复错误： 网络超时、临时限流。
- 用户输入错误： 日期格式错误、缺少参数。
- 业务错误： 数据不存在、用户无权限。
- 程序错误： 代码 Bug、未预期的异常。
不要把所有错误都简单地转换成“查询失败”，否则后续排查问题会很困难。
## 八、今天的实战：开发多工具电价分析 Agent
下面给你一份可以交给 AI Coding 实现的完整项目要求。建议先让它生成代码、运行测试，再逐个阅读工具和执行流程。
请开发一个基于 LangChain 的电价分析 Agent 学习项目，项目名为 price_analysis_agent。
### 8.1 技术要求
- Python
- LangChain 当前版本，使用 langchain.agents.create_agent
- @tool 定义工具
- Pydantic 定义参数和报告结构
- 使用环境变量配置模型 API
- 使用 pytest 编写测试
请勿使用已经过时的 Agent 初始化方式作为新项目主方案。依赖版本必须明确，并以实际安装版本的官方 API 为准。
### 8.2 Agent 功能
用户可以用自然语言提出电价分析任务，例如：

“分析 2026-10-26 的实时电价预测效果，计算 MAE 和 RMSE，并找出误差最大的两个时段。”

Agent 需要根据任务调用工具，最终输出结构化分析报告。
### 8.3 实现三个工具
工具一：get_price_data
输入：
- date
- market：DA 或 RT
功能：
查询指定日期和市场的预测电价、实际电价及时间点。

第一版使用明确标注的模拟数据，不要伪装成真实市场数据。模拟数据至少包含 4 个时段，并包含一定程度的预测误差。
工具二：calculate_error_metrics
输入：
- date
- market
功能：
查询相应数据，使用 Python 计算 MAE 和 RMSE。

公式：
MAE = 误差绝对值的平均值。
RMSE = 误差平方平均值的平方根。

不要让大模型自己估算指标。
工具三：find_worst_hours
输入：
- date
- market
- top_n
功能：
按绝对预测误差从大到小排序，返回误差最大的若干时段，以及预测值、实际值和绝对误差。
### 8.4 Agent 要求
1. 使用 create_agent。
2. 配置清晰的 system prompt。
3. Agent 能根据用户任务调用工具。
4. 不能编造不存在的电价数据。
5. 日期没有数据时，要明确报告数据不足。
6. 使用 Pydantic 定义最终报告，包含状态、日期、市场、MAE、RMSE、误差最大的时段及分析结论。
7. 如果模型支持，使用 response_format 返回结构化报告。
8. 设置模型 API 请求超时和有限重试。
9. 设置合理的执行步数限制。
10. 打印工具调用过程、工具结果和最终报告，方便学习和调试。
### 8.5 工程要求
项目至少包含：
- main.py
- tools.py
- schemas.py
- config.py
- tests/
- requirements.txt
- .env.example
- README.md
不得将真实 API 密钥写入源码。

请为三个工具编写测试，覆盖正常输入、日期不存在、非法参数等情况。测试工具时应尽量使用模拟数据，不依赖真实大模型 API。
### 8.6 完成后请解释
1. 每个文件的作用。
2. 模型如何选择工具。
3. 工具调用请求如何变成实际 Python 执行。
4. 工具结果如何返回模型。
5. 结构化报告如何生成。
6. 如何验证 MAE 和 RMSE 的计算正确。
7. 当前实现距离生产环境还有哪些差距。
请先给出项目结构和实现方案，再编写代码。完成后运行测试，并根据实际测试结果报告问题，不要声称未经执行的测试已经通过。

## 九、工具调用的完整消息链

工具调用不是模型直接执行 Python 函数，而是模型和程序之间的消息往返：

```text
HumanMessage：用户提出任务
        ↓
AIMessage：模型返回 tool_calls
        ↓
程序：校验工具名和参数，执行 Python 函数
        ↓
ToolMessage：返回工具结果和 tool_call_id
        ↓
AIMessage：模型生成最终回答或继续调用工具
```

可以手动执行这一流程：

```python
from langchain_core.messages import HumanMessage, ToolMessage

model_with_tools = model.bind_tools(tools)
messages = [HumanMessage(content="请计算 100 加 200")]

ai_message = model_with_tools.invoke(messages)
messages.append(ai_message)

for call in ai_message.tool_calls:
    selected_tool = tool_map[call["name"]]
    tool_result = selected_tool.invoke(call["args"])
    messages.append(ToolMessage(
        content=str(tool_result),
        tool_call_id=call["id"],
    ))

final_message = model_with_tools.invoke(messages)
print(final_message.content)
```

`tool_call_id` 用来把工具结果和模型发出的具体调用对应起来。一次模型响应可以包含多个工具调用，程序应逐个校验和执行，或根据业务规则限制并发数量。

## 十、结构化输出与工具调用的区别

两者解决的问题不同：

| 能力 | 目的 | 典型结果 |
| --- | --- | --- |
| Tool Calling | 让模型请求程序执行动作 | 工具名、参数、调用 id |
| Structured Output | 让模型最终返回固定数据结构 | Pydantic 对象或 JSON |

一个 Agent 可以先调用工具，再将工具结果整理成结构化报告。结构化输出只约束字段和类型，不能替代 Python 对 MAE、RMSE 等指标的真实计算。

## 十一、Chain、Router 和 Agent 的区别

**Chain** 的步骤由程序固定，适合稳定流程：

```text
输入 → 提示词 → 模型 → 结构化输出
```

**Router** 先判断问题类型，再选择不同链：

```text
问题分类 → 技术问答链 / 数据分析链 / 翻译链
```

**Agent** 允许模型在运行时选择工具和下一步动作，适合工具数量多、任务路径不固定的场景。

能够用固定 Chain 完成的任务，通常不必使用 Agent。固定流程更容易测试、监控和控制成本。

## 十二、测试、调试和生产化检查

开发时建议分别验证每一层：

1. 单独调用模型，确认 API、模型名称和消息格式正确；
2. 单独调用工具，确认参数和业务逻辑正确；
3. 打印并检查 `AIMessage.tool_calls`；
4. 检查 `ToolMessage.tool_call_id` 是否正确对应；
5. 测试工具失败、参数错误和空结果；
6. 测试 Agent 达到最大步数时是否能够安全结束；
7. 使用日志或 LangSmith 查看完整链路和 token 消耗；
8. 验证模型版本、依赖版本和配置文件可复现。

生产环境还需要重点考虑：

- 工具权限和用户身份校验；
- 数据库使用参数化查询，禁止模型直接拼接任意 SQL；
- 外部 API 的超时、限流、重试和缓存；
- 有副作用工具的幂等性和人工确认；
- Agent 最大执行步数、总耗时和 token 预算；
- 日志脱敏、调用审计、错误告警和结果追踪；
- 用固定测试集验证指标计算和结构化输出。

## 十三、版本和导入说明

LangChain 生态更新较快，模型、工具和 Agent 的高层 API 可能随版本变化。建议把核心包版本固定在项目依赖文件中，并以当前安装版本的官方文档为准。常见包包括：

```text
langchain
langchain-core
langchain-openai
langchain-community
langgraph
```

对于新代码，优先使用 `langchain_core` 中的 Runnable、消息和工具抽象，避免混用旧版 Agent API。升级依赖后，应重新运行工具调用、结构化输出和异常处理测试。

## 十四、总结

LangChain 应用通常由以下部分组成：

```text
模型 Model
提示词 Prompt
消息 Message
Runnable/Chain
工具 Tool
检索器 Retriever
Agent/Graph
```

工具调用的本质是“模型提出结构化调用请求，应用执行真实函数，再把结果交回模型”。模型负责选择和组织信息，程序负责执行、校验、权限控制和最终安全边界。理解这一点后，就能区分普通 Chain、工具调用和 Agent，并根据流程是否固定选择合适的实现方式。

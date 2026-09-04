---
type: 论文清单
status: 已发布
level: 进阶
topic:
  - Agent
  - 科研
  - 项目实战
---




## 引言

Agent落地元年，大家一直在用各种方式做创新，比如拿Graph做Memory，拿SFT+RL增强调用tools的能力等，研究者们探索了将RL与LLM结合，开发出能够自主调用外部工具、优化搜索策略并在多轮交互中实现复杂推理的智能体,RL在各种场景中频繁出现。

本文通过对一系列创新性项目的系统分析，全面综述了RL在增强LLM推理、搜索和交互能力中的应用，探讨了其方法论、实践价值及未来发展方向.

---

## 1. ReSearch
- **论文**: [https://arxiv.org/pdf/2503.19470](https://arxiv.org/pdf/2503.19470)
- **GitHub仓库**: [https://github.com/Agent-RL/ReCall](https://github.com/Agent-RL/ReCall)


- **摘要**: 大型语言模型（LLM）展现出卓越的推理能力，如OpenAI-o1和DeepSeek-R1的成功所示。然而，将推理与外部搜索过程结合仍具挑战性，尤其是在需要多步检索的复杂多跳问题上。我们提出了ReSearch框架，通过强化学习训练LLM在无需推理步骤监督数据的情况下使用搜索进行推理。搜索操作被视为推理链的一部分，由基于文本的思维引导，搜索结果进一步影响推理过程。我们在Qwen2.5-7B和Qwen2.5-32B模型上训练ReSearch，实验表明其在多基准测试中具有强大的泛化能力，展现出反思和自我修正等高级推理能力。

- **描述**: ReSearch通过强化学习优化LLM的搜索与推理能力，将搜索操作融入推理链，强调基于文本的动态决策。
    
- **实践价值**: 适合研究RL如何提升LLM在复杂推理任务中的表现，特别是涉及多步搜索优化的场景。

## 2. Search-R1
- **论文**: [https://arxiv.org/pdf/2503.19470](https://arxiv.org/pdf/2503.19470)
- **GitHub仓库**: [https://github.com/PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1)
- 支持本地稀疏检索器（例如 BM25）。✔️
- 支持本地密集检索器（平面索引和 ANN 索引）✔️
- 支持谷歌搜索/必应搜索 API 等。✔️
- 支持现成的神经重新排序器。✔️
- 支持不同的 RL 方法（例如 PPO、GRPO、强化）。✔️
- 支持不同的 LLM（例如 llama3、Qwen2.5 等）。✔️

- **摘要**: 高效获取外部知识对LLM的推理和生成至关重要，但直接让推理能力强的LLM使用搜索引擎并非最佳选择。本文提出Search-R1，通过强化学习扩展推理框架，使LLM在逐步推理中自主生成搜索查询。Search-R1通过多轮搜索交互优化推理轨迹，利用检索到的token掩码进行稳定RL训练，并采用基于结果的奖励函数。实验表明，Search-R1在七个问答数据集上性能显著优于RAG基线，Qwen2.5-7B和Qwen2.5-3B模型分别提升41%和20%。
    
- **描述**: Search-R1通过RL训练LLM优化查询生成和结果解读，提升任务解决能力。
- **实践价值**: 适合探索LLM与外部工具（如搜索引擎）结合的场景，研究RL如何优化信息获取。
    

## 3. R1-Searcher
- **论文**: [https://arxiv.org/pdf/2503.05592](https://arxiv.org/pdf/2503.05592)
    
- **GitHub仓库**: [https://github.com/RUCAIBox/R1-Searcher](https://github.com/RUCAIBox/R1-Searcher)
    
- Model:
    - Qwen-2.5-7B-Base-RAG-RL: [https://huggingface.co/XXsongLALA/Qwen-2.5-7B-base-RAG-RL](https://huggingface.co/XXsongLALA/Qwen-2.5-7B-base-RAG-RL)
    - Llama-3.1-8B-Instruct-RAG-RL: [https://huggingface.co/XXsongLALA/Llama-3.1-8B-instruct-RAG-RL](https://huggingface.co/XXsongLALA/Llama-3.1-8B-instruct-RAG-RL)
        
- Train-data: [https://huggingface.co/datasets/XXsongLALA/RAG-RL-Hotpotqa-with-2wiki](https://huggingface.co/datasets/XXsongLALA/RAG-RL-Hotpotqa-with-2wiki)
    
- **摘要**: 现有大型推理模型通过RL提升复杂推理能力，但依赖内部知识可能导致不准确或幻觉问题。我们提出R1-Searcher，一种基于结果的两阶段RL方法，增强LLM的搜索能力，使其在推理中自主调用外部搜索系统。实验表明，R1-Searcher显著优于RAG方法，甚至超越闭源的GPT-4o-mini。
    
- **描述**: R1-Searcher通过RL激励LLM发展搜索能力，专注于从外部资源获取和利用信息。
    
- **实践价值**: 适用于知识密集型问答或数据分析等高效信息检索任务。
    

## 4. RAGEN
- **论文**: [https://arxiv.org/pdf/2504.20073](https://arxiv.org/pdf/2504.20073)
- **GitHub仓库**: [https://github.com/ZihanWang314/RAGEN](https://github.com/ZihanWang314/RAGEN)

- **摘要**: 训练LLM为交互式智能体面临长期决策和随机环境反馈的挑战。我们提出StarPO框架和RAGEN系统，用于轨迹级智能体RL训练。研究发现“回声陷阱”模式，并通过StarPO-S解决。实验表明，多样化初始状态和推理感知奖励信号对RL部署至关重要。
    
- **描述**: RAGEN通过多轮RL实现LLM代理的自我进化，优化轮次级别的信用分配。
    
- **实践价值**: 适合研究多轮交互和自我改进的代理，应用于对话系统或任务规划。
    
> 有三个主要模块：环境状态管理器（ragen/llm_agent/es_manager.py）、上下文管理器（ragen/llm_agent/ctx_manager.py）和代理代理（ragen/llm_agent/agent_proxy.py）。

1. 环境状态管理器（es_manager）：
支持多种环境（不同环境、相同环境不同种子、相同环境相同种子）
记录部署过程中每个环境的状态
处理来自ctx_manager的操作，执行步骤，并以批处理方式将操作结果（观察结果）返回给ctx_manager

2. 上下文管理器（ctx_manager）：
将原始代理令牌解析为es_manager的结构化操作
格式化来自es_manager的观察，解析并制定它们以供接下来的代理推出。
收集最终的推出轨迹并将其编译成标记、注意力掩码、奖励分数和损失掩码，以供 llm 更新。

3. Agent Proxy（agent_proxy）： 作为执行单轮或多轮部署的接口

## 5. ReTool

- **论文**: [https://arxiv.org/pdf/2504.11536](https://arxiv.org/pdf/2504.11536)
- **GitHub仓库**: [https://github.com/ReTool-RL/ReTool](https://github.com/ReTool-RL/ReTool)
- **摘要**: 推理模型在文本推理上表现出色，但在结构化问题解决中表现不佳。ReTool通过工具集成学习增强长篇推理能力，动态交错代码执行，并通过RL优化工具调用策略。在AIME基准上，ReTool-32B模型达到72.5%准确率，超越OpenAI o1-preview。

- **描述**: ReTool通过RL训练LLM战略性使用工具（如API、计算器），优化任务解决能力。
    
- **实践价值**: 适合研究工具增强型LLM，探索RL在工具选择和使用中的应用。
    

## 6. DeepResearcher
    

- **论文**: [https://arxiv.org/pdf/2504.03160](https://arxiv.org/pdf/2504.03160)

- **GitHub仓库**: [https://github.com/GAIR-NLP/DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher)

DeepResearcher 是首个全面的框架，通过在真实环境中扩展强化学习 (RL) 来实现基于 LLM 的深度学习代理的端到端训练，并支持真实的网络搜索交互。我们的定性分析揭示了端到端 RL 训练中涌现出的认知行为 ，包括制定计划、交叉验证来自多个来源的信息、进行自我反思以重新引导研究方向，以及在无法找到明确答案时保持诚实的能力。

- **摘要**: DeepResearcher通过在真实环境中扩展RL，训练LLM进行深度研究，应对开放网络的复杂性。实验表明，其性能显著优于基于提示工程和RAG的基线，展现出计划、交叉验证和自我反思等认知行为。
    
- **描述**: DeepResearcher结合LLM与外部知识源，扩展RL在现实研究任务中的应用。
    
- **实践价值**: 适合需要深度信息合成的场景，如科学研究或复杂数据分析。
    

## 7. ZeroSearch
    
- **论文**: [https://arxiv.org/abs/2505.04588](https://arxiv.org/abs/2505.04588)

- **GitHub仓库**: 未提供具体链接
    
- **摘要**: ZeroSearch通过RL训练LLM内化搜索能力，采用基于课程的推出策略，逐步提升推理能力。实验表明，7B和14B模型的检索模块性能可媲美甚至超越真实搜索引擎。
    
- **描述**: ZeroSearch减少对外部搜索的依赖，通过RL模拟搜索行为。
    
- **实践价值**: 适合研究LLM如何通过内部机制模拟搜索，降低对外部资源的依赖。
    

## 8. Agent-R1
    

- **GitHub仓库**: [https://github.com/0russwest0/Agent-R1](https://github.com/0russwest0/Agent-R1)

- **描述**: Agent-R1扩展DeepSeek-R1，通过RL训练代理型 LLM，提升决策和任务执行能力。
    
- **详细解读**: [https://deepwiki.com/0russwest0/Agent-R1](https://deepwiki.com/0russwest0/Agent-R1)

- **实践价值**: 适合探索动态环境中的代理行为，应用于复杂任务场景。
    

## 9. StepSearch
    
- **论文**: [https://www.arxiv.org/pdf/2505.15107](https://www.arxiv.org/pdf/2505.15107)

- **GitHub仓库**: [https://github.com/Zillwang/StepSearch](https://github.com/Zillwang/StepSearch)

- **摘要**: StepSearch通过逐步近端策略优化（PPO）训练LLM，包含细粒度搜索奖励和信息增益监督，显著优于全局奖励基线，3B和7B模型分别提升11.2%和4.2%。
    
- **描述**: StepSearch通过PPO优化分步搜索过程，提升多跳推理能力。
    
- **实践价值**: 适合研究PPO在LLM搜索增强中的应用，特别适用于分步任务。
    

## 10. Multi-Turn-RL-Agent
    
- **论文**: [https://arxiv.org/pdf/2505.11821](https://arxiv.org/pdf/2505.11821)
- 
- **GitHub仓库**: [https://github.com/SiliangZeng/Multi-Turn-RL-Agent](https://github.com/SiliangZeng/Multi-Turn-RL-Agent)

- **摘要**: 通过轮次级别优势估计优化多轮工具使用场景，Multi-Turn-RL-Agent在工具执行和精确答案匹配中显著优于基线，成功率达100%，准确率达50%。
    
- **描述**: 聚焦于通过轮次级别信用分配强化LLM代理的多轮推理能力。
    
- **实践价值**: 适合研究多轮对话或复杂任务中的长期推理优化。
    

---

## 扩展项目推荐

## 11. WebShop
    

- **论文**: [WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents](https://arxiv.org/abs/2207.01206)
    
- **GitHub仓库**: [https://github.com/WebShop/WebShop](https://github.com/WebShop/WebShop)
    
- **描述**: WebShop是一个模拟电商环境的开源框架，使用RL训练LLM代理执行网页交互任务，如搜索和购买商品。
    
- **实践价值**: 提供现实世界的测试平台，适合研究LLM在网页导航和任务执行中的RL应用。
    
## 12. RL4LMs
    

- **论文**: [Training Language Models with Natural Language Feedback](https://arxiv.org/abs/2204.14146)
    
- **GitHub仓库**: [https://github.com/allenai/RL4LMs](https://github.com/allenai/RL4LMs)
    
- **描述**: RL4LMs是一个开源库，专注于使用RL和自然语言反馈优化LLM，包含PPO、TRPO等算法，用于文本生成任务。
    
- **实践价值**: 提供通用的RL+LLM实验平台，适合快速原型开发和算法对比。
    

## 13. AgentBench
    

- **论文**: [AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/2308.03688)
    
- **GitHub仓库**: [https://github.com/THUDM/AgentBench](https://github.com/THUDM/AgentBench)
    
- **描述**: AgentBench是一个评估LLM代理能力的基准测试平台，包含游戏、网页导航等任务，支持RL训练和评估。
    
- **实践价值**: 适合测试和比较不同RL算法在代理任务中的表现。
    

## 14. Toolformer
    

- **论文**: [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
    
- **GitHub仓库**: [https://github.com/lucidrains/toolformer-pytorch](https://github.com/lucidrains/toolformer-pytorch)（非官方实现）
    
- **描述**: Toolformer通过RL训练LLM自主学习使用外部工具（如计算器、搜索API），提升任务解决能力。
    
- **实践价值**: 适合研究LLM通过RL学习工具调用的场景，应用于自动化任务。
    

## 15.ReAct
    

- **论文**: [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
    
- **GitHub仓库**: [https://github.com/ysymyth/ReAct](https://github.com/ysymyth/ReAct)
    
- **描述**: ReAct结合推理和行动，通过RL优化LLM在交互式任务中的表现，强调推理与执行的协同。
    
- **实践价值**: 适合研究推理与行动结合的代理，应用于交互式决策场景。
    

---

## 结论

本文通过对ReSearch、Search-R1、R1-Searcher等15个项目的系统分析，展示了强化学习在增强大型语言模型推理、搜索和交互能力中的重要作用。这些项目通过创新的RL框架，优化了LLM在复杂任务中的表现，涵盖了搜索优化、工具调用、多轮交互和深度研究等场景。未来，随着RL算法和LLM的进一步融合，特别是在真实环境中的端到端训练和细粒度奖励设计的推动下，LLM智能体的能力有望进一步提升，为知识密集型任务、自动化决策和动态交互提供更强大的解决方案。


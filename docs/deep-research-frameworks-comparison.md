# Deep Research 框架技术调研报告

## Executive Summary

本报告深入对比分析了当前主流的8个Deep Research开源框架，并与PandaDive进行全面对比。通过架构分析、功能对比和最佳实践研究，为PandaDive的未来演进提供清晰的优化路线图。

**关键发现：**
- PandaDive在架构设计上已具备较强的竞争力，三层图结构清晰
- 主要差距在于评估体系、缓存层和可观测性
- 建议优先实施评估框架和缓存层，可显著提升用户体验

---

## 目录

1. [主流框架概览](#主流框架概览)
2. [PandaDive架构深度解析](#pandadive架构深度解析)
3. [全面对比矩阵](#全面对比矩阵)
4. [优化建议与路线图](#优化建议与路线图)
5. [实施计划](#实施计划)

---

## 主流框架概览

### 1. Firecrawl Deep Research ⭐ 最成熟

**项目地址：** [firecrawl/firecrawl](https://github.com/firecrawl/firecrawl)

**技术架构：**
```
Deep Research Service
├── Research State Manager (Redis)
│   ├── Activities Tracking
│   ├── Findings Aggregation
│   └── Depth Control
├── Research Executor
│   ├── Web Crawling
│   ├── Content Extraction
│   └── LLM Analysis
└── Storage Layer
    ├── Redis Cache
    └── GCS Persistence
```

**核心特点：**
- **完整生命周期管理**：从研究初始化到结果输出的全流程管控
- **深度控制机制**：通过 maxDepth (1-10) 精确控制研究深度
- **活动类型多样化**：search、extract、analyze、reasoning 等多种活动
- **高性能缓存层**：Redis + GCS 双层存储架构
- **企业级特性**：完整 API、SDK 支持、监控告警

**代码示例：**
```typescript
// Firecrawl Deep Research API
const research = await firecrawl.deepResearch({
  query: "Quantum computing latest developments",
  maxDepth: 7,
  maxUrls: 15,
  activities: ["search", "extract", "analyze"]
});
```

---

### 2. LangChain Open Deep Research ⭐ 官方推荐

**项目地址：** [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research)

**技术架构：**
```
Research Pipeline
├── Query Generation
├── Parallel Search
├── Source Evaluation
├── Content Synthesis
└── Report Generation

Evaluation Framework
├── Pairwise Comparison
├── Completeness Scoring
├── Source Quality Assessment
└── Citation Verification
```

**核心特点：**
- **完善的评估框架**：业界最完整的评估体系
  - Pairwise evaluation：对比两个研究结果的优劣
  - Completeness scoring：评估回答完整性
  - Source quality assessment：来源质量评分
  - Citation verification：引用验证
- **LangGraph 原生**：深度集成 LangGraph 生态
- **模块化设计**：各组件可独立使用和替换
- **研究深度可控**：通过配置调整研究粒度

**评估示例：**
```python
# Pairwise Evaluation
from open_deep_research.evaluation import pairwise_evaluate

result = pairwise_evaluate(
    question="What are the latest AI safety measures?",
    answer_a=research_result_a,
    answer_b=research_result_b,
    criteria=[
        "completeness",
        "source_diversity", 
        "citation_quality"
    ]
)
```

---

### 3. Tongyi Deep Research (阿里巴巴)

**项目地址：** [Alibaba-NLP/DeepResearch](https://github.com/Alibaba-NLP/DeepResearch)

**技术架构：**
```
Multi-Agent Framework
├── Planning Agent
│   ├── Task Decomposition
│   └── Strategy Selection
├── Execution Agent
│   ├── Web Search
│   ├── Document Retrieval
│   └── Code Execution
└── Synthesis Agent
    ├── Information Fusion
    ├── Conflict Resolution
    └── Report Generation
```

**核心特点：**
- **多智能体协作**：Planning、Execution、Synthesis 三类 Agent 协同
- **中文优化**：针对中文语境的深度优化
- **多源融合**：Web 搜索 + 文档检索 + 代码执行
- **阿里云生态**：与阿里云产品深度集成
- **开源领先**：阿里 NLP 团队开源贡献

---

### 4. DeerFlow (字节跳动)

**项目地址：** [bytedance/deer-flow](https://github.com/bytedance/deer-flow)

**技术架构：**
```
Community-Driven Framework
├── Tool Integration Layer
│   ├── Web Search
│   ├── Web Crawling
│   └── Python Execution
├── Workflow Engine
│   ├── Node Definition
│   ├── Edge Routing
│   └── State Management
└── Community Extensions
    ├── Plugin System
    └── Template Library
```

**核心特点：**
- **社区驱动**：开源社区贡献模式和插件生态
- **工具丰富**：搜索、爬取、Python 执行三位一体
- **工作流引擎**：灵活的节点定义和边路由
- **模板系统**：可复用的研究模板库
- **字节技术栈**：字节跳动技术团队开源

---

### 5. Onyx (企业级)

**项目地址：** [onyx-dot-app/onyx](https://github.com/onyx-dot-app/onyx)

**技术架构：**
```
Enterprise Q&A Platform
├── Multi-Source Connectors
│   ├── Document Repositories
│   ├── Knowledge Bases
│   └── External APIs
├── Research Agent
│   ├── Internal Search
│   ├── Web Search
│   └── Synthesis
└── Enterprise Features
    ├── Access Control
    ├── Audit Logging
    └── Analytics
```

**核心特点：**
- **企业级问答**：面向企业知识管理的完整平台
- **多源连接器**：支持 20+ 种数据源集成
- **内外结合**：内部知识库 + 外部 Web 搜索
- **权限管控**：企业级访问控制和审计
- **生产就绪**：完整的部署和运维支持

---

### 6. Local Deep Researcher (完全本地)

**项目地址：** [langchain-ai/local-deep-researcher](https://github.com/langchain-ai/local-deep-researcher)

**技术架构：**
```
Fully Local Pipeline
├── Local LLM (Ollama/llama.cpp)
├── Local Search (DuckDuckGo/SearXNG)
├── Local Storage (SQLite/Chroma)
└── Privacy-First Design
```

**核心特点：**
- **完全本地**：无需外部 API，保护隐私
- **离线可用**：无网络环境下可运行
- **轻量级**：资源占用低，适合个人设备
- **LangChain 官方**：LangChain 团队维护

---

### 7. OpenAI Agents SDK Deep Research

**项目地址：** [qx-labs/agents-deep-research](https://github.com/qx-labs/agents-deep-research)

**技术架构：**
```
OpenAI Agents SDK Implementation
├── Agent Definition
│   ├── Instructions
│   └── Tools
├── Handoffs
│   ├── Research Agent
│   ├── Analysis Agent
│   └── Writing Agent
└── Guardrails
    └── Output Validation
```

**核心特点：**
- **OpenAI Agents SDK**：基于最新的 Agents SDK
- **迭代式研究**：多轮迭代深化研究
- **现代架构**：最新的 Agent 设计模式
- **易于扩展**：清晰的 Agent 定义和切换

---

## PandaDive架构深度解析

### 三层图结构

PandaDive采用清晰的三层架构设计：

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Graph (主图)                        │
├─────────────────────────────────────────────────────────────┤
│  START → clarify_with_user → write_research_brief          │
│                                    ↓                       │
│                    research_supervisor ─────────┐          │
│                           ↓                      │          │
│                final_report_generation ←─────────┘          │
│                           ↓                                │
│                          END                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Supervisor Subgraph (监督者子图)                 │
├─────────────────────────────────────────────────────────────┤
│  START → supervisor ──→ supervisor_tools                   │
│              ↑                │                              │
│              └──────────────┘ (循环直到完成)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Researcher Subgraph (研究员子图)                 │
├─────────────────────────────────────────────────────────────┤
│  START → researcher ──→ researcher_tools                  │
│              ↑                │                              │
│              └──────────────┘ (REACT 循环)                  │
│                           ↓                                │
│                    compress_research                       │
│                           ↓                                │
│                          END                               │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件详解

#### 1. 主图（Main Graph）

**clarify_with_user（需求澄清）**
- **职责**: 分析用户输入，判断是否需要澄清问题
- **输出**: ClarifyWithUser结构化输出（need_clarification, question, verification）
- **特点**: 支持结构化输出和JSON解析降级

**write_research_brief（生成简报）**
- **职责**: 将用户需求转化为结构化研究简报
- **输出**: ResearchQuestion（research_brief字段）
- **作用**: 为监督者提供清晰的研究目标和范围

**research_supervisor（研究监督者）**
- **职责**: 监督者子图的入口，协调整个研究过程
- **特点**: 动态委派研究任务，支持并发执行

**final_report_generation（生成最终报告）**
- **职责**: 整合所有研究发现，生成结构化报告
- **输入**: notes（所有研究发现）
- **输出**: final_report（Markdown格式）

#### 2. 监督者子图（Supervisor Subgraph）

**supervisor（监督者节点）**
- **职责**: 战略规划，决定如何委派研究任务
- **可用工具**: 
  - ConductResearch（委派研究任务）
  - ResearchComplete（标记研究完成）
  - think_tool（战略思考）
- **退出条件**:
  - 超过最大迭代次数
  - 无工具调用
  - 研究完成工具调用

**supervisor_tools（监督者工具节点）**
- **职责**: 执行监督者调用的工具
- **并发执行**: 支持最多 `max_concurrent_research_units` 个研究任务并行
- **处理流程**:
  1. 检查退出条件
  2. 处理 think_tool 调用
  3. 并行执行 ConductResearch 研究任务
  4. 更新监督者消息和状态

#### 3. 研究员子图（Researcher Subgraph）

**researcher（研究员节点）**
- **职责**: 执行具体的研究任务
- **REACT 模式**: 推理（Reasoning）→ 行动（Action）→ 观察（Observation）
- **可用工具**:
  - Search 工具（Tavily/DuckDuckGo/ArXiv）
  - think_tool（研究反思）
  - MCP 工具（外部扩展）
- **退出条件**:
  - 超过最大工具调用次数
  - 无工具调用
  - ResearchComplete 调用

**researcher_tools（研究员工具节点）**
- **职责**: 执行研究员工具调用
- **检索质量循环**（已实施）:
  1. 查询重写（Query Rewrite）
  2. 搜索执行
  3. 结果解析
  4. 相关性评分
  5. 结果重排序
  6. 质量指标记录

**compress_research（研究压缩节点）**
- **职责**: 压缩和综合研究结果
- **输入**: researcher_messages（所有研究消息）
- **输出**: compressed_research（结构化摘要）
- **处理逻辑**:
  1. 准备压缩提示
  2. 调用压缩模型
  3. 处理 token 限制错误（截断重试）
  4. 返回压缩结果和原始笔记

### 状态管理详解

#### 状态节点继承关系

```
MessagesState (LangGraph基础)
    ↓
AgentInputState (输入状态)
    ↓
AgentState (主图状态)
    ├── supervisor_messages
    ├── research_brief
    ├── raw_notes
    ├── notes
    └── final_report
    ↓
SupervisorState (监督者状态)
    └── research_iterations
    
ResearcherState (研究员状态)
    ├── researcher_messages
    ├── research_topic
    ├── tool_call_iterations
    ├── compress_research
    ├── rewritten_queries
    ├── relevance_scores
    ├── reranked_results
    └── quality_notes
```

#### Reducer 机制

**override_reducer**: 用于覆盖状态值
```python
def override_reducer(current_value, new_value):
    """状态节点reducer."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)
```

**使用场景**:
- `supervisor_messages`: 覆盖消息列表
- `research_brief`: 覆盖研究简报
- `compressed_research`: 覆盖压缩结果

---

## 全面对比矩阵

### 功能对比表

| 功能维度 | PandaDive | Firecrawl | LangChain Open | Tongyi | DeerFlow | Onyx | Local Researcher | OpenAI Agents |
|---------|-----------|-----------|----------------|--------|----------|------|------------------|---------------|
| **架构模式** | 三层图 | 迭代循环 | 图+评估 | 多Agent | 多Agent | 混合 | 单机 | 简单图 |
| **并发能力** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| **研究深度控制** | 迭代数 | maxDepth+活动 | 迭代数 | 深度+策略 | 迭代+工具 | 深度+策略 | 无 | 简单循环 |
| **检索质量** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| **评估框架** | ❌ | ❌ | ⭐⭐⭐⭐⭐ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **缓存层** | ❌ | ⭐⭐⭐⭐⭐ | ❌ | ⭐⭐ | ❌ | ⭐⭐⭐⭐ | ❌ | ❌ |
| **可观测性** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ |
| **企业特性** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ |
| **中文支持** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐ |
| **社区活跃度** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |

### 技术栈对比

| 技术维度 | PandaDive | Firecrawl | LangChain Open | Tongyi | DeerFlow | Onyx | Local Researcher | OpenAI Agents |
|---------|-----------|-----------|----------------|--------|----------|------|------------------|---------------|
| **基础框架** | LangGraph | Node.js/Bull | LangGraph | 自研 | 自研 | FastAPI | LangChain | Agents SDK |
| **LLM支持** | 多供应商 | OpenAI优先 | 多供应商 | 阿里云 | 多供应商 | 多供应商 | Ollama | OpenAI |
| **搜索API** | Tavily/DuckDuckGo | 内置爬取 | Tavily/Exa | 阿里搜索 | 多源 | 20+连接器 | DuckDuckGo | Web Search |
| **存储** | 内存 | Redis+GCS | 内存 | 分布式 | 内存 | Postgres+Vespa | SQLite | 内存 |
| **缓存** | ❌ | Redis | ❌ | 有限 | ❌ | 多层级 | ❌ | ❌ |
| **流式输出** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐ |
| **API接口** | Python | REST+SDK | Python | Python | Python | REST | Python | Python |
| **部署方式** | 库 | 云服务/自托管 | 库 | 云服务 | 库 | 企业部署 | 本地 | 库 |

---

## 优化建议与路线图

### 核心差距分析

```
差距严重程度矩阵

                    影响程度
                    低    中    高
                  ┌─────┬─────┬─────┐
        高        │     │     │     │
    实  中   难  │缓存 │评估 │企业 │
    现  低   度  │层   │框架 │特性 │
                  ├─────┼─────┼─────┤
        高        │     │     │     │
    影  中   易  │网页 │可观 │MCP  │
    响  低   度  │浏览 │测性 │扩展 │
                  └─────┴─────┴─────┘
```

### P0 - 核心架构优化（优先级：最高 ⭐⭐⭐⭐⭐）

#### 1. 评估框架（最大差距）

**现状问题：**
- 无法量化研究质量
- 无法对比不同研究策略
- 无法自动优化参数

**目标架构：**
```python
# evaluation/framework.py
class ResearchEvaluationFramework:
    """研究评估框架 - 类似LangChain Open的评估体系"""
    
    def __init__(self):
        self.evaluators = {
            'completeness': CompletenessEvaluator(),
            'accuracy': AccuracyEvaluator(),
            'source_quality': SourceQualityEvaluator(),
            'citation_quality': CitationEvaluator(),
            'readability': ReadabilityEvaluator()
        }
    
    async def evaluate_report(
        self,
        research_brief: str,
        final_report: str,
        sources: List[Source],
        criteria: List[str] = None
    ) -> EvaluationResult:
        """综合评估研究报告"""
        
    async def pairwise_comparison(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        criteria: List[str]
    ) -> PairwiseResult:
        """两两对比评估 - 核心功能"""
```

**关键组件：**

| 评估器 | 功能 | 指标 |
|-------|------|------|
| CompletenessEvaluator | 评估回答完整性 | coverage_score, gap_analysis |
| AccuracyEvaluator | 验证事实准确性 | fact_check_score, hallucination_rate |
| SourceQualityEvaluator | 评估来源质量 | domain_authority, citation_count |
| CitationEvaluator | 验证引用质量 | citation_accuracy, source_diversity |
| ReadabilityEvaluator | 评估可读性 | complexity_score, structure_score |

**实施步骤：**
1. 创建 `evaluation/` 目录
2. 实现基础评估器基类
3. 实现具体评估器
4. 集成到主图中（final_report_generation 后）
5. 添加配置选项（启用/禁用评估）

**预期效果：**
- 量化研究质量
- 支持A/B测试不同策略
- 自动优化参数配置

---

#### 2. 缓存层（性能关键）

**现状问题：**
- 重复查询浪费token
- 无法利用历史研究结果
- 每次都要重新搜索

**目标架构：**
```python
# cache/manager.py
class ResearchCacheManager:
    """研究缓存管理器 - 多级缓存架构"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.caches = self._initialize_caches()
        
    def _initialize_caches(self) -> Dict[str, CacheBackend]:
        """初始化多级缓存"""
        return {
            'memory': MemoryCache(max_size=1000, ttl=300),  # 5分钟
            'disk': DiskCache(cache_dir='.cache/panda_dive', ttl=3600),  # 1小时
            'redis': RedisCache(redis_url=os.getenv('REDIS_URL')) if os.getenv('REDIS_URL') else None
        }
    
    async def get_research_result(
        self, 
        query: str, 
        config: dict
    ) -> Optional[Dict]:
        """获取缓存的研究结果"""
        cache_key = self._generate_cache_key(query, config)
        
        for cache in self.caches.values():
            if cache:
                result = await cache.get(cache_key)
                if result:
                    return result
        
        return None
    
    async def set_research_result(
        self, 
        query: str, 
        config: dict, 
        result: Dict
    ):
        """缓存研究结果"""
        cache_key = self._generate_cache_key(query, config)
        
        for cache in self.caches.values():
            if cache:
                await cache.set(cache_key, result)
    
    def _generate_cache_key(self, query: str, config: dict) -> str:
        """生成查询缓存键"""
        # 包含查询内容和关键配置
        content = f"{query}:{json.dumps(config, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]
```

**缓存策略：**

| 层级 | 后端 | TTL | 容量 | 用途 |
|------|------|-----|------|------|
| L1 | Memory | 5分钟 | 1000条 | 热点数据 |
| L2 | Disk (SQLite) | 1小时 | 10GB | 单机持久化 |
| L3 | Redis | 24小时 | 分布式 | 集群共享 |

**实施步骤：**
1. 创建 `cache/` 目录结构
2. 实现 `CacheBackend` 抽象基类
3. 实现 `MemoryCache`、`DiskCache`、`RedisCache`
4. 实现 `ResearchCacheManager`
5. 集成到 `researcher_tools` 中
6. 添加配置选项

**预期效果：**
- 减少 30-50% 的重复搜索
- 降低 token 消耗
- 提升响应速度

---

## 实施计划

### Phase 1: 基础建设（1-2周）

#### Week 1: 评估框架基础
- [ ] 创建 `evaluation/` 目录结构
- [ ] 实现 `BaseEvaluator` 抽象类
- [ ] 实现 `CompletenessEvaluator`
- [ ] 添加基础测试

#### Week 2: 缓存层基础
- [ ] 创建 `cache/` 目录结构
- [ ] 实现 `CacheBackend` 抽象类
- [ ] 实现 `MemoryCache`
- [ ] 实现 `DiskCache`
- [ ] 添加基础测试

### Phase 2: 核心功能（2-3周）

#### Week 3-4: 评估框架完善
- [ ] 实现所有评估器
- [ ] 集成到主图中
- [ ] 添加配置选项
- [ ] 完善测试

#### Week 4-5: 缓存层完善
- [ ] 实现 `RedisCache`
- [ ] 实现 `ResearchCacheManager`
- [ ] 集成到 researcher_tools
- [ ] 完善测试

### Phase 3: 工程化（1-2周）

#### Week 6: 可观测性
- [ ] 添加详细日志
- [ ] 实现性能指标收集
- [ ] 添加研究耗时统计

#### Week 7: 文档和示例
- [ ] 完善技术文档
- [ ] 添加使用示例
- [ ] 创建配置模板

### 预期成果

#### 性能提升
- 减少 30-50% 的重复搜索
- 降低 20-30% 的 token 消耗
- 提升 20-40% 的响应速度

#### 质量提升
- 可量化的研究质量评分
- 支持A/B测试不同策略
- 自动优化参数配置

#### 用户体验
- 更稳定的性能
- 更快的响应速度
- 更透明的研究过程

---

## 附录

### 参考资源

1. **Firecrawl Deep Research**
   - 项目地址: https://github.com/firecrawl/firecrawl
   - API文档: https://docs.firecrawl.dev/deep-research

2. **LangChain Open Deep Research**
   - 项目地址: https://github.com/langchain-ai/open_deep_research
   - 评估框架: https://github.com/langchain-ai/open_deep_research/tree/main/evaluation

3. **Tongyi Deep Research**
   - 项目地址: https://github.com/Alibaba-NLP/DeepResearch
   - 论文: https://arxiv.org/abs/2401.12345

4. **LangGraph 文档**
   - https://langchain-ai.github.io/langgraph/

5. **PandaDive 项目**
   - 项目地址: https://github.com/123yongming/Panda_Dive
   - 文档: https://github.com/123yongming/Panda_Dive/tree/main/docs

### 术语表

| 术语 | 英文 | 说明 |
|------|------|------|
| 深度研究 | Deep Research | 多轮迭代、多源融合的智能研究过程 |
| 监督者 | Supervisor | 负责协调和委派研究任务的Agent |
| 研究员 | Researcher | 执行具体研究任务的Agent |
| 检索质量 | Retrieval Quality | 搜索结果的准确性和相关性 |
| 评估框架 | Evaluation Framework | 量化研究质量的评估体系 |
| 缓存层 | Cache Layer | 存储历史研究结果的多级缓存 |
| 可观测性 | Observability | 追踪和监控系统运行状态的能力 |
| REACT模式 | REACT Pattern | 推理-行动-观察的循环模式 |
| MCP | Model Context Protocol | 模型上下文协议，用于扩展工具 |

---

**文档版本**: v1.0  
**创建日期**: 2025-02-02  
**最后更新**: 2025-02-02  
**维护者**: PandaDive Team

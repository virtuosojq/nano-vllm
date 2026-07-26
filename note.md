# 阶段 1 笔记：为什么 `LLM` 类只有一个 `pass`？

## 代码

```python
# nanovllm/llm.py
from nanovllm.engine.llm_engine import LLMEngine

class LLM(LLMEngine):
    pass
```

`LLM` 是空壳，所有逻辑（`generate` / `add_request` / `step`）都在父类 `LLMEngine`。

## 设计原因

1. **关注点分离**：`LLMEngine` = 引擎实现（调度/序列/TP/KV cache），`LLM` = 对外接口名。
2. **便于引擎层复用**：内部模块可直接引用 `LLMEngine`，避免反向依赖对外 API。
3. **为扩展留口子**：未来可在 `LLM` 上加用户便利方法（`chat()` / `stream_generate()`），不污染核心。
4. **符合 vLLM 惯例**：复刻 vLLM 的命名（用户熟悉的 `LLM`），实现放在 `LLMEngine`。
5. **测试友好**：可独立操作 `LLMEngine` 做测试/mock。

## 反面对比

把所有逻辑塞进 `LLM` 会导致：类过大、内部方法与用户接口混杂、引擎无独立名字可被引用。

## 一句话总结

> `LLM` 是门面，`LLMEngine` 是引擎。`pass` 表示门面暂未做额外包装，纯粹借名给引擎用——**门面模式（Facade）** 的简化形态。

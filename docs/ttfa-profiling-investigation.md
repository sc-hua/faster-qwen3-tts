# TTFA Profiling 调查记录

记录 2026-04-15 对 faster-qwen3-tts 首包延迟 (TTFA) 波动问题的完整调查过程。
平台：NVIDIA Jetson AGX Thor (tegra264, aarch64)。

## 现象

Web 客户端报告 TTFA 在 300ms ~ 2400ms 之间波动。
相同文本连续请求 TTFA 稳定 ~300ms，换文本或间隔一段时间后飙到 2s+。

## 调查路径

### 1. 外层定位：prepare_generation vs prefill vs decode

在 `generate_voice_clone_streaming()` 中加计时：

```python
import time
t0 = time.monotonic()
m, talker, config, tie, tam, tth, tpe, ref_codes = self._prepare_generation(...)
t_prep = time.monotonic() - t0
logger.debug("prepare_generation: %.1fms | tie=%s tth=%s", t_prep * 1000, tie.shape, tth.shape)
```

结果：`prepare_generation: 2031ms`，而 prefill 和 decode 都正常。问题在准备阶段。

### 2. 中层定位：tokenize vs resolve_vcp vs build_talker

在 `_prepare_generation()` 内部拆分三段计时：

```python
# tokenize: _build_assistant_text + _tokenize_texts
# resolve_vcp: _resolve_voice_clone_prompt
# build_talker: _build_talker_inputs_local
```

结果：`tokenize=2029.7ms resolve_vcp=0.0ms build_talker=2.8ms`。瓶颈在 tokenize。

### 3. 细粒度定位：CPU tokenizer vs .to(cuda)

进一步拆分 tokenize 步骤：

```python
t0 = time.monotonic()
text = model._build_assistant_text('hello')        # 拼字符串
t1 = time.monotonic()
out = processor(text=text, return_tensors='pt')     # HF tokenizer (CPU)
t2 = time.monotonic()
ids = out['input_ids'].to('cuda:0')                 # CPU -> GPU
torch.cuda.synchronize()
t3 = time.monotonic()
```

结果（空闲 15s 后）：
- `build_text`: 0.0ms
- `processor()`: 1.3ms
- `.to(cuda)`: **2027.9ms** ← 全部延迟在这里

### 4. 验证：裸 tensor .to(cuda) 也慢

```python
x = torch.randint(0, 1000, (1, 10))  # CPU 上的小 tensor
# ... warm up ...
time.sleep(15)
y = x.to('cuda:0')  # 2028.1ms
```

确认：跟 tokenizer 无关，任何 CPU→GPU 操作在空闲后都要 ~2s。

### 5. 排除假设

| 假设 | 验证 | 结论 |
|------|------|------|
| 内存 swap/page cache | `free -h`: 84G 空闲, swap=0 | ❌ 排除 |
| GPU 降频 | `jetson_clocks --show`: gpu 315MHz | 但锁频到 1575MHz 后依然 2s |
| GPU 降频是根因 | `sudo jetson_clocks` 锁最高频 | ❌ 排除，锁频无效 |
| CUDA context 休眠 | 加 keepalive 心跳 | ✅ 有效，TTFA 从 2.3s→0.36s |

### 6. 根因

Jetson Thor 的 CUDA runtime 在空闲 ~10-15s 后进入某种休眠/省电状态。
第一次 GPU 操作（哪怕只是一个 10 元素 tensor 的 `.to(cuda)`）需要 ~2s 唤醒。
这不是 GPU 时钟频率问题（锁频无效），而是 CUDA driver/runtime 层面的行为。

### 7. 解决方案

在 API server 的 lifespan 中添加 CUDA keepalive 心跳：

```python
_KEEPALIVE_INTERVAL = 5  # seconds
_keepalive_tensor = torch.zeros(1, device=device)

async def _cuda_keepalive():
    while True:
        await asyncio.sleep(_KEEPALIVE_INTERVAL)
        _keepalive_tensor.add_(1)
```

每 5 秒对 GPU tensor 做一次 add_ 操作，保持 CUDA context 活跃。

### TTFA 对比

| 场景 | 空闲 70s 后 TTFA |
|------|-----------------|
| 无 keepalive + 无锁频 | 2.34s |
| 无 keepalive + 锁频 | 2.17s |
| **keepalive** + 无锁频 | **0.36s** |

## Profiling 方法论总结

1. **外层到内层**：先在最外层函数加计时，定位到哪个子调用慢，再逐层深入。
2. **加 `torch.cuda.synchronize()`**：GPU 操作是异步的，不 sync 的 timing 会把前一个操作的延迟错误计入当前操作。之前看到 `prepare_generation: 2031ms` 的假象就是因为缺少 sync。
3. **独立复现**：用 `docker exec` 在容器内单独跑 Python 脚本，排除 API server 的并发/队列因素。
4. **控制变量**：连续请求 vs 空闲后请求，相同文本 vs 不同文本，锁频 vs 动态频率。
5. **裸操作验证**：用最简单的操作（`torch.randint().to(cuda)`）验证是不是特定函数的问题还是底层 runtime 的问题。

## _build_talker_inputs_local 内部计时参考

如需再次 profiling build_talker，子步骤拆分点：
- `generate_speaker_prompt(voice_clone_prompt)` — x-vector 提取
- `text_projection(get_text_embeddings()(input_id))` — 文本嵌入投影
- `generate_icl_prompt(...)` — ICL 模式的参考音频编码（包含 thinker forward）

注意每个 checkpoint 前加 `torch.cuda.synchronize()` 才能得到准确的 GPU 时间。

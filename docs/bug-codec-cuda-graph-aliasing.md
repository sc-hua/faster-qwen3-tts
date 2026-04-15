# Bug: Codec CUDA Graph 跨 Size 内存别名导致音频永久性失真

**日期**: 2026-04-15
**组件**: `faster_qwen3_tts/codec_graph.py` → `CodecGraphDecoder`
**平台**: aarch64 Jetson Thor

## 现象

- 并发的非流式 + 流式 TTS 请求后，**流式输出永久性爆音**（peak=1.0 饱和削波），非流式始终正常
- 一旦触发，后续所有流式请求都会产出失真音频，直到进程重启
- 纯流式并发或纯非流式并发不会触发
- 服务端日志无任何 error/warning，无 OOB tokens，无 NaN

## 复现条件

1. API server 启动并完成 CUDA graph warmup
2. 同时发出一个非流式请求和一个流式请求（通过 HTTP 并发即可）
3. 流式响应即出现爆音，后续流式请求永久失真

复现脚本：

```python
import subprocess, numpy as np, threading

TEXT = "欢迎来到我们的直播间，今天给大家带来一款超值好物。"

def req(idx, stream):
    mode = "stream" if stream else "nostream"
    d = '{"input":"' + TEXT + '","stream":' + str(stream).lower() + ',"response_format":"pcm","chunk_size":4,"voice":"english"}'
    subprocess.run(["curl", "-4", "-s", "-m", "120", "-X", "POST",
        "http://127.0.0.1:23456/v1/audio/speech",
        "-H", "Content-Type: application/json", "-d", d,
        "-o", f"/tmp/tts_{idx}.pcm"], capture_output=True)
    data = np.frombuffer(open(f"/tmp/tts_{idx}.pcm", "rb").read(), dtype=np.int16)
    audio = data.astype(np.float32) / 32768.0
    peak = float(np.max(np.abs(audio)))
    print(f"  {idx} {mode:8s}: peak={peak:.4f}" + (" *** NOISY ***" if peak > 0.9 else ""))

# 并发：非流式 + 流式
t1 = threading.Thread(target=req, args=(1, False))
t2 = threading.Thread(target=req, args=(2, True))
t1.start(); t2.start()
t1.join(); t2.join()
```

## 根因

`CodecGraphDecoder` 原实现为 7 个不同 capture size `[4, 8, 12, 25, 29, 37, 50]` 各自 capture 了独立的 CUDA graph。

问题在于：这些 graph 共享同一个 decoder 模型实例，PyTorch 在 graph capture 时会为中间计算结果分配 GPU 内存地址。**不同 size 的 graph 的中间 buffer 可能被分配到相同的 GPU 内存地址**（memory aliasing）。

当 replay `graph[50]`（非流式 decode 路径）时，该 graph 的中间计算会写入这些共享地址。随后 replay `graph[29]` 或更小的 graph（流式 decode 路径）时，它期望这些地址保持 capture 时的初始状态，但实际已被 `graph[50]` 覆盖。结果 decoder 网络在错误的中间状态上继续计算，产出数值极端的波形。

由于 CUDA graph capture 时内存布局是确定性的，一旦污染发生就是永久性的——后续所有 replay 小 graph 都会读到同样的脏数据。

## 为什么非流式不受影响

非流式 decode 一次处理完整序列（通常 50+ 帧），始终命中最大 size 的 `graph[50]`。`graph[50]` 只与自身 replay，不存在跨 graph 冲突。

## 排除的方案：`graph_pool_handle()`

`torch.cuda.graph_pool_handle()` 创建的是一个共享内存池——它的设计目的是让多个 graph **故意共享**内存以节省显存，而不是隔离。每次调用 `graph_pool_handle()` 返回的是新 pool，但问题不在 pool 层面：同一个 decoder 模型的权重和中间 buffer 在所有 graph 中是同一份物理内存。

实测该方案无效——部署后 10 轮并发全部爆音。

## 修复

改为只 capture **一个** `max_size=50` 的 CUDA graph。所有输入统一零填充到 50 帧后 replay，输出按实际长度裁剪。

```python
class CodecGraphDecoder:
    def __init__(self, decoder, max_size=50):
        self.decoder = decoder
        self.max_size = max_size
        self.graph = None
        self.static_input = None
        self.static_output = None

    def capture(self, device, num_warmup=3):
        # 只 capture 一个 max_size 的 graph
        graph = CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = self.decoder(static_input)
        self.graph = graph
        ...

    def __call__(self, codes):
        actual = codes.shape[-1]
        if self.graph is None or actual > self.max_size:
            return self.decoder(codes)
        self.static_input.zero_()
        self.static_input[:, :, :actual] = codes
        self.graph.replay()
        return self.static_output[..., :actual * self.total_upsample].clone()
```

- 只有一个 graph，不再存在跨 graph 内存别名
- 零填充区域的计算开销可忽略（decoder 是 CNN，计算量与帧数线性相关）
- 30 轮并发压力测试（60 请求）验证零爆音

## 验证结果

| 测试场景 | 修复前 | 修复后 |
|---------|-------|-------|
| 纯流式顺序 10 次 | ✅ 正常 | ✅ 正常 |
| 纯非流式顺序 10 次 | ✅ 正常 | ✅ 正常 |
| 并发流式×2 | ✅ 正常 | ✅ 正常 |
| 并发非流式×2 | ✅ 正常 | ✅ 正常 |
| 并发非流式+流式 1 次 | ❌ 流式爆音 | ✅ 正常 |
| 并发非流式+流式 30 轮 | ❌ 立即永久失真 | ✅ 60/60 正常 |

## 关联问题

测试过程中还观察到偶发的服务卡死（事件循环阻塞），与此 bug 可能有关联但属于独立问题，需要单独排查 `_stream_tts` 的异步生成器生命周期管理。

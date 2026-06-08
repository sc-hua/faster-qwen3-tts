# faster-qwen3-tts API Server

这个仓库当前面向 `api_server.py` 使用，提供一个基于 FastAPI 的 Qwen3-TTS 服务端。

主要能力：

- `POST /v1/audio/speech`：语音合成
- `GET /v1/audio/voices`：列出已注册音色
- `GET /v1/audio/voices/{name}`：查看单个音色
- `POST /v1/audio/voices`：上传参考音频并注册持久化音色
- `DELETE /v1/audio/voices/{name}`：删除音色

## 环境要求

- Python 3.10+
- CUDA GPU
- 建议使用 `bfloat16`

安装依赖：

```bash
pip install -r requirements.txt
```

## 启动

默认启动：

```bash
python api_server.py
```

指定模型和端口：

```bash
python api_server.py \
  --model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --device cuda:0 \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000 \
  --voices_dir voices
```

主要参数：

- `--model_path`：模型路径或 Hugging Face Hub ID
- `--device`：推理设备，默认 `cuda:0`
- `--dtype`：默认 `bfloat16`
- `--host`：监听地址
- `--port`：监听端口
- `--voices_dir`：音色缓存和持久化目录
- `--default_ttl`：运行时缓存音色默认过期时间，单位秒

## 接口说明

### 1. 语音合成

`POST /v1/audio/speech`

请求体是 JSON。必须提供 `voice` 或 `ref_audio` 二选一。

常用字段：

- `input`：要合成的文本
- `voice`：已注册音色名
- `ref_audio`：参考音频，支持：
  - `data:audio/...;base64,...`
  - `http(s)://...`
  - `file:///abs/path`
  - 本地绝对路径
- `ref_text`：参考音频文本；当 `xvec_only=false` 时必填
- `language`：默认 `Auto`
- `xvec_only`：默认 `true`
  - `true`：x-vector 模式，速度更快
  - `false`：ICL 模式，质量更高，但必须提供 `ref_text`
- `stream`：是否流式返回，默认 `false`
- `response_format`：`wav`、`pcm`、`flac`、`mp3`
- `max_new_tokens`
- `repetition_penalty`
- `instruct`：风格或口音提示
- `chunk_size`：流式模式下每块 codec 帧数

简单示例，直接用本地参考音频：

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "你好，这是一个测试。",
    "ref_audio": "file:///absolute/path/ref.wav",
    "xvec_only": true,
    "response_format": "wav"
  }' \
  --output out.wav
```

使用已注册音色：

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "你好，这是一个测试。",
    "voice": "demo_voice",
    "response_format": "wav"
  }' \
  --output out.wav
```

ICL 模式：

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "你好，这是 ICL 模式测试。",
    "ref_audio": "file:///absolute/path/ref.wav",
    "ref_text": "参考音频对应的文本",
    "xvec_only": false,
    "response_format": "wav"
  }' \
  --output out.wav
```

流式模式：

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "这是流式合成测试。",
    "ref_audio": "file:///absolute/path/ref.wav",
    "stream": true,
    "response_format": "wav",
    "chunk_size": 4
  }' \
  --output out.wav
```

注意：

- 流式返回实际只支持 `wav` 和 `pcm`，传其他格式时会退回 `pcm`
- 返回头里会包含 `X-Sample-Rate` 和 `X-Request-Id`
- 服务端会串行化 GPU 推理请求，不会并发跑多个生成任务
- CUDA Graph 加速路径固定使用 `temperature=0.9`、`top_k=50`、
  `top_p=1.0` 和采样模式，HTTP API 不提供运行时采样参数

### 2. 音色管理

列出音色：

```bash
curl http://127.0.0.1:8000/v1/audio/voices
```

查看音色：

```bash
curl http://127.0.0.1:8000/v1/audio/voices/demo_voice
```

创建音色：

```bash
curl http://127.0.0.1:8000/v1/audio/voices \
  -F name=demo_voice \
  -F mode=xvec \
  -F ref_text='' \
  -F audio_sample=@/absolute/path/ref.wav
```

创建 ICL 音色：

```bash
curl http://127.0.0.1:8000/v1/audio/voices \
  -F name=demo_voice_icl \
  -F mode=icl \
  -F ref_text='参考音频对应的文本' \
  -F audio_sample=@/absolute/path/ref.wav
```

删除音色：

```bash
curl -X DELETE http://127.0.0.1:8000/v1/audio/voices/demo_voice
```

约束：

- 音色名长度 1 到 64
- 必须以字母或数字开头
- 只能包含 `a-z`、`A-Z`、`0-9`、`_`、`-`
- 上传音频最大 10MB
- 参考音频时长必须在 1 到 30 秒之间
- `mode=icl` 时必须提供 `ref_text`

## 存储说明

`--voices_dir` 默认是 `voices/`，内部会保存：

- `registry.json`：音色元数据
- `persistent/`：持久化音色 prompt
- `cache/`：运行时缓存 prompt

服务启动时会清理过期或残留的运行时缓存，并做一次 warmup。

## 错误返回

接口错误统一返回：

```json
{
  "error": {
    "message": "错误信息",
    "type": "invalid_request_error",
    "code": 400
  }
}
```

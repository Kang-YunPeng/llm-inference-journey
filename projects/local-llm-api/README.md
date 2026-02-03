# Local LLM API

OpenAI兼容的本地LLM推理API

## 快速开始

### 方法1：使用启动脚本（推荐）

```bash
# 安装依赖
pip install -r requirements.txt

# 一键启动（自动检查并下载模型）
./start.sh
```

### 方法2：手动启动

```bash
# 安装依赖
pip install -r requirements.txt

# 下载模型
python src/download_qwen.py

# 运行
python src/main.py
```

## API使用

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen1.5-0.5b-chat",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

## Docker

```bash
docker build -t local-llm-api .
docker run -p 8000:8000 local-llm-api
```
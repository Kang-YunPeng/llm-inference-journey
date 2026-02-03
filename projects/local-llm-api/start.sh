#!/bin/bash
# =============================================================================
# start.sh - 自动检查模型并启动服务
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
MODEL_PATH="$PROJECT_ROOT/models/Qwen--Qwen1.5-0.5B-Chat"
DOWNLOAD_SCRIPT="$PROJECT_ROOT/src/download_qwen.py"
MAIN_SCRIPT="$PROJECT_ROOT/src/main.py"

echo "=========================================="
echo "  Local LLM API - 启动脚本"
echo "=========================================="
echo ""

# === 检查模型是否存在 ===
if [ -d "$MODEL_PATH" ]; then
    echo "✅ 模型已存在: $MODEL_PATH"
else
    echo "⚠️  模型不存在，开始下载..."
    echo "📁 模型路径: $MODEL_PATH"
    echo ""
    
    if [ ! -f "$DOWNLOAD_SCRIPT" ]; then
        echo "❌ 下载脚本不存在: $DOWNLOAD_SCRIPT"
        exit 1
    fi
    
    python3 "$DOWNLOAD_SCRIPT"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 模型下载完成"
    else
        echo ""
        echo "❌ 模型下载失败"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "  启动服务..."
echo "=========================================="
echo ""

# === 启动服务 ===
python3 "$MAIN_SCRIPT"

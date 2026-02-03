#!/bin/bash
# =============================================================================
# generate_requirements.sh —— 生成锁定的 requirements.txt（使用官方 PyTorch 源）
# 支持从任意目录调用，自动定位项目根目录
# =============================================================================

set -euo pipefail

# === 安全获取脚本所在目录和项目根目录 ===
# 无论通过 ./script.sh, bash script.sh, 或绝对路径调用，都能正确解析
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"  # 确保是绝对路径

REQUIREMENTS_IN="$PROJECT_ROOT/requirements.in"
REQUIREMENTS_OUT="$PROJECT_ROOT/requirements.txt"
VENV_NAME=".venv-reqgen"

# PyTorch 官方 CUDA 12.1 wheel 索引
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"

# 配置
PIP_VERSION="23.3.2"
PYTHON_MIN_VERSION="3.12"

# === 日志函数 ===
log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $*" >&2; }
error() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2; exit 1; }

# === 启动日志 ===
log "🚀 开始生成锁定依赖文件"
log "📁 脚本目录: $SCRIPT_DIR"
log "🏠 项目根目录: $PROJECT_ROOT"
log "📄 requirements.in: $REQUIREMENTS_IN"

# === 验证 requirements.in 存在且非空 ===
if [[ ! -f "$REQUIREMENTS_IN" ]]; then
    error "requirements.in 未找到！请确保在项目根目录运行此脚本。"
fi

if [[ ! -s "$REQUIREMENTS_IN" ]]; then
    error "requirements.in 为空！"
fi

# === 检查 Python 版本 ===
log "🔍 检查系统环境..."
python3 -c "import sys; v=sys.version_info; assert v.major==3 and v.minor>=12, f'Need Python>=3.12, got {v.major}.{v.minor}'" \
    || error "需要 Python >= 3.12"

log "✅ 环境检查通过 (Python $(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'))"

# # === 创建临时虚拟环境 ===
log "🔄 创建临时虚拟环境: $VENV_NAME"
rm -rf "$VENV_NAME"
python3 -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

# # === 升级 pip ===
log "🛠️  安装 pip==$PIP_VERSION..."
python3 -m pip install --upgrade "pip==$PIP_VERSION" --quiet

# # === 安装依赖（使用官方 PyTorch 源）===
log "📥 安装所有依赖（使用官方 PyTorch 源）..."

# # 关键：直接使用已验证存在的文件
python3 -m pip install \
    --extra-index-url "$PYTORCH_INDEX_URL" \
    -r "$REQUIREMENTS_IN" \
    --timeout 1000 \
    --retries 5 \
    --progress-bar on \
    || error "依赖安装失败，请检查网络或 requirements.in 内容"

# === 验证 PyTorch CUDA 可用性（可选）===
if python3 -c 'import torch' &>/dev/null; then
    if python3 -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
        log "✅ CUDA 可用 (PyTorch $(python3 -c 'import torch; print(torch.__version__)'))"
    else
        log "⚠️  CUDA 不可用（继续生成锁定文件）"
    fi
else
    log "⚠️  PyTorch 未安装成功（继续生成锁定文件）"
fi

# === 生成锁定文件 ===
log "🔒 生成 $REQUIREMENTS_OUT..."
{
    echo "# Auto-generated lock file (official PyTorch source)"
    echo "# $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
    echo ""
    python3 -m pip freeze
} > "$REQUIREMENTS_OUT"

# # === 清理 ===
deactivate || true
rm -rf "$VENV_NAME"

log "✅ 成功生成 requirements.txt"
log "📄 输出路径: $REQUIREMENTS_OUT"
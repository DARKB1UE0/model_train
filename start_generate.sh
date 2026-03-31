#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLEND_FILE="${1:-${PROJECT_DIR}/first.blend}"
PY_SCRIPT="${PROJECT_DIR}/generate_ore_data.py"

if ! command -v blender >/dev/null 2>&1; then
  echo "错误: 未找到 blender 命令，请先安装 Blender。"
  exit 1
fi

if [[ ! -f "${BLEND_FILE}" ]]; then
  echo "错误: 未找到 .blend 文件: ${BLEND_FILE}"
  exit 1
fi

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "错误: 未找到 Python 脚本: ${PY_SCRIPT}"
  exit 1
fi

echo "启动 Blender 后台渲染..."
echo "场景文件: ${BLEND_FILE}"
echo "脚本文件: ${PY_SCRIPT}"

exec blender -b "${BLEND_FILE}" -P "${PY_SCRIPT}"

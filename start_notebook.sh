#!/bin/bash

echo "🚀 Jupyter Notebook 启动脚本"
echo "=" * 30

# 检查是否在正确目录
TARGET_DIR="/Users/xiang/Documents/GitHub/NewStart/NewStart"
if [ "$PWD" != "$TARGET_DIR" ]; then
    echo "📁 切换到项目目录..."
    cd "$TARGET_DIR" || {
        echo "❌ 无法找到目录: $TARGET_DIR"
        exit 1
    }
fi

echo "✅ 当前目录: $(pwd)"

# 检查Jupyter是否安装
if ! command -v jupyter &> /dev/null; then
    echo "❌ Jupyter未安装，正在安装..."
    pip install jupyter notebook
fi

# 检查端口8888是否被占用
if lsof -Pi :8888 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  端口8888被占用，使用端口8889"
    jupyter notebook --port=8889
else
    echo "🚀 启动Jupyter Notebook..."
    jupyter notebook
fi
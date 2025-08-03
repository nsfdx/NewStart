#!/usr/bin/env python3
# Jupyter Notebook 启动器
import subprocess
import sys

def start_notebook():
    try:
        # 尝试不同的启动方式
        commands = [
            [sys.executable, '-m', 'notebook', '--port=8888'],
            [sys.executable, '-m', 'jupyter', 'notebook', '--port=8888'], 
            ['jupyter', 'notebook', '--port=8888'],
            ['jupyter-notebook', '--port=8888']
        ]
        
        for cmd in commands:
            try:
                print(f"尝试: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
                break
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                print(f"失败: {e}")
                continue
        else:
            print("所有启动方式都失败了")
            
    except KeyboardInterrupt:
        print("\n服务器已停止")

if __name__ == "__main__":
    start_notebook()

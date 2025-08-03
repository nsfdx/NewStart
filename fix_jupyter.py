import subprocess
import sys
import os
import platform

def check_python_environment():
    """检查Python环境"""
    print("🐍 Python环境检查:")
    print(f"  Python版本: {sys.version}")
    print(f"  Python路径: {sys.executable}")
    print(f"  操作系统: {platform.system()}")
    print()

def install_jupyter_comprehensive():
    """全面安装Jupyter"""
    print("🔧 开始全面安装Jupyter...")
    
    commands = [
        # 升级pip
        [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
        # 安装jupyter
        [sys.executable, '-m', 'pip', 'install', 'jupyter'],
        # 安装notebook
        [sys.executable, '-m', 'pip', 'install', 'notebook'],
        # 安装jupyterlab (可选，更现代的界面)
        [sys.executable, '-m', 'pip', 'install', 'jupyterlab'],
    ]
    
    for i, cmd in enumerate(commands, 1):
        try:
            print(f"步骤 {i}: 执行 {' '.join(cmd[-2:])}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ 步骤 {i} 完成")
        except subprocess.CalledProcessError as e:
            print(f"❌ 步骤 {i} 失败: {e}")
            print(f"错误输出: {e.stderr}")
            return False
    
    return True

def check_jupyter_commands():
    """检查所有可能的Jupyter命令"""
    commands_to_try = [
        'jupyter',
        'jupyter-notebook', 
        'jupyter-lab',
        f'{sys.executable} -m jupyter',
        f'{sys.executable} -m notebook'
    ]
    
    print("🔍 检查可用的Jupyter命令:")
    working_commands = []
    
    for cmd in commands_to_try:
        try:
            if cmd.startswith(sys.executable):
                # 对于Python模块命令，需要特殊处理
                cmd_parts = cmd.split()
                result = subprocess.run(cmd_parts + ['--version'], 
                                      capture_output=True, text=True, timeout=10)
            else:
                result = subprocess.run([cmd, '--version'], 
                                      capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print(f"  ✅ {cmd}: 可用")
                working_commands.append(cmd)
            else:
                print(f"  ❌ {cmd}: 不可用")
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
            print(f"  ❌ {cmd}: 未找到")
    
    return working_commands

def start_jupyter_alternative():
    """使用替代方法启动Jupyter"""
    working_commands = check_jupyter_commands()
    
    if not working_commands:
        print("❌ 没有找到可用的Jupyter命令")
        return False
    
    # 优先使用的命令顺序
    preferred_order = [
        'jupyter notebook',
        f'{sys.executable} -m notebook',
        'jupyter-notebook',
        f'{sys.executable} -m jupyter notebook',
        'jupyter lab'
    ]
    
    for preferred in preferred_order:
        for working in working_commands:
            if working in preferred or preferred in working:
                try:
                    print(f"🚀 使用命令启动: {preferred}")
                    
                    # 处理端口
                    port = 8888
                    if not check_port_availability(port):
                        port = 8889
                    
                    if preferred.startswith(sys.executable):
                        cmd = preferred.split() + [f'--port={port}', '--no-browser']
                    else:
                        cmd = preferred.split() + [f'--port={port}', '--no-browser']
                    
                    print(f"📝 执行命令: {' '.join(cmd)}")
                    print(f"🌐 请在浏览器中访问: http://localhost:{port}")
                    print("按 Ctrl+C 停止服务器")
                    
                    subprocess.run(cmd)
                    return True
                    
                except KeyboardInterrupt:
                    print("\n👋 Jupyter服务器已停止")
                    return True
                except Exception as e:
                    print(f"❌ 启动失败: {e}")
                    continue
    
    return False

def check_port_availability(port=8888):
    """检查端口是否可用"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except OSError:
            return False

def create_notebook_launcher():
    """创建notebook启动脚本"""
    launcher_content = f"""#!/usr/bin/env python3
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
                print(f"尝试: {{' '.join(cmd)}}")
                subprocess.run(cmd, check=True)
                break
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                print(f"失败: {{e}}")
                continue
        else:
            print("所有启动方式都失败了")
            
    except KeyboardInterrupt:
        print("\\n服务器已停止")

if __name__ == "__main__":
    start_notebook()
"""
    
    launcher_path = "start_notebook.py"
    with open(launcher_path, 'w', encoding='utf-8') as f:
        f.write(launcher_content)
    
    print(f"✅ 创建了启动脚本: {launcher_path}")
    print(f"可以运行: python {launcher_path}")

def main():
    print("🚀 Jupyter Notebook 问题诊断和修复工具")
    print("=" * 60)
    
    # 1. 检查Python环境
    check_python_environment()
    
    # 2. 检查当前目录
    current_dir = os.getcwd()
    target_dir = "/Users/xiang/Documents/GitHub/NewStart/NewStart"
    
    if current_dir != target_dir and os.path.exists(target_dir):
        os.chdir(target_dir)
        print(f"📁 切换到目录: {target_dir}")
    
    print(f"📍 当前工作目录: {os.getcwd()}")
    print()
    
    # 3. 尝试安装Jupyter
    print("🔧 检查和安装Jupyter...")
    if not install_jupyter_comprehensive():
        print("❌ Jupyter安装失败，但继续尝试其他方法")
    
    print()
    
    # 4. 检查可用命令
    working_commands = check_jupyter_commands()
    print()
    
    # 5. 创建启动脚本
    create_notebook_launcher()
    print()
    
    # 6. 尝试启动
    if working_commands:
        print("🚀 尝试启动Jupyter Notebook...")
        start_jupyter_alternative()
    else:
        print("❌ 无法启动Jupyter，请尝试以下解决方案:")
        print("1. 重新安装Python和pip")
        print("2. 使用conda安装: conda install jupyter notebook")
        print("3. 使用虚拟环境重新安装")
        print("4. 检查系统PATH环境变量")

if __name__ == "__main__":
    main()
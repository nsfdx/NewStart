import subprocess
import sys
import os

def check_jupyter_installation():
    """检查Jupyter是否正确安装"""
    try:
        result = subprocess.run(['jupyter', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Jupyter已安装")
            print(f"版本信息:\n{result.stdout}")
            return True
        else:
            print("❌ Jupyter未正确安装")
            return False
    except FileNotFoundError:
        print("❌ Jupyter未安装")
        return False

def install_jupyter():
    """安装Jupyter Notebook"""
    print("🔧 正在安装Jupyter Notebook...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'jupyter', 'notebook'], 
                      check=True)
        print("✅ Jupyter安装完成")
        return True
    except subprocess.CalledProcessError:
        print("❌ Jupyter安装失败")
        return False

def check_port_availability(port=8888):
    """检查端口是否可用"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            print(f"✅ 端口 {port} 可用")
            return True
        except OSError:
            print(f"❌ 端口 {port} 被占用")
            return False

def start_jupyter_safe():
    """安全启动Jupyter"""
    # 检查安装
    if not check_jupyter_installation():
        if not install_jupyter():
            return False
    
    # 检查端口
    port = 8888
    if not check_port_availability(port):
        port = 8889
        print(f"🔄 尝试使用端口 {port}")
    
    # 启动Jupyter
    try:
        print(f"🚀 启动Jupyter Notebook在端口 {port}...")
        cmd = ['jupyter', 'notebook', f'--port={port}']
        subprocess.run(cmd)
        return True
    except KeyboardInterrupt:
        print("\n👋 Jupyter已停止")
        return True
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Jupyter Notebook 诊断和修复工具")
    print("=" * 50)
    
    # 检查当前目录
    current_dir = os.getcwd()
    target_dir = "/Users/xiang/Documents/GitHub/NewStart/NewStart"
    
    if current_dir != target_dir:
        try:
            os.chdir(target_dir)
            print(f"📁 切换到目录: {target_dir}")
        except FileNotFoundError:
            print(f"❌ 目录不存在: {target_dir}")
            print(f"当前在: {current_dir}")
    
    # 启动Jupyter
    start_jupyter_safe()
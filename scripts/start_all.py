"""
一键启动医疗知识问答系统（后端API + 前端Streamlit）
"""

import subprocess
import sys
import os
import time
import signal
import atexit

# 项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # scripts的父目录就是项目根目录
# 确保使用绝对路径
PROJECT_ROOT = os.path.normpath(PROJECT_ROOT)
VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
VENV_UVICORN = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "uvicorn.exe")
VENV_STREAMLIT = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "streamlit.exe")

# 存储启动的进程
processes = []


def get_venv_python():
    """获取虚拟环境的Python解释器路径"""
    # 使用当前解释器路径，更稳定
    return sys.executable


def get_frontend_path():
    """获取前端应用路径，确保使用正斜杠"""
    # 使用正斜杠格式，确保跨平台兼容
    return os.path.join(PROJECT_ROOT, "app", "main.py").replace("\\", "/")


def get_venv_executable(name):
    """获取虚拟环境中的可执行文件路径"""
    if os.name == "nt":  # Windows
        return os.path.join(PROJECT_ROOT, ".venv", "Scripts", f"{name}.exe")
    else:  # Unix
        return os.path.join(PROJECT_ROOT, ".venv", "bin", name)


def start_backend():
    """启动后端API服务"""
    print("\n" + "=" * 50)
    print("🚀 启动后端API服务...")
    print("=" * 50)
    
    # 切换到项目根目录
    os.chdir(PROJECT_ROOT)
    
    # 使用python -m uvicorn 方式启动，更兼容
    cmd = [get_venv_python(), "-m", "uvicorn", "backend.api.main:app", "--reload", "--port", "8000"]
    
    # 创建新进程组，在Windows下使用 CREATE_NEW_PROCESS_GROUP
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        creationflags = 0
    
    process = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        creationflags=creationflags
    )
    
    processes.append(("backend", process))
    
    # 等待服务启动
    print("⏳ 等待后端服务启动...")
    time.sleep(3)
    
    # 检查进程是否还在运行
    if process.poll() is not None:
        print("❌ 后端服务启动失败！")
        output, _ = process.communicate()
        print(output)
        return False
    
    print("✅ 后端API服务已启动: http://localhost:8000")
    print("   API文档: http://localhost:8000/docs")
    return True


def start_frontend():
    """启动Streamlit前端"""
    print("\n" + "=" * 50)
    print("🚀 启动Streamlit前端...")
    print("=" * 50)
    
    # 切换到项目根目录
    os.chdir(PROJECT_ROOT)
    
    # 获取前端应用路径
    app_path = get_frontend_path()
    
    # 使用python -m streamlit 方式启动
    cmd = [get_venv_python(), "-m", "streamlit", "run", app_path, "--server.port", "8501"]
    
    # 创建新进程组
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        creationflags = 0
    
    process = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        creationflags=creationflags
    )
    
    processes.append(("frontend", process))
    
    # 等待服务启动
    print("⏳ 等待前端服务启动...")
    time.sleep(3)
    
    # 检查进程是否还在运行
    if process.poll() is not None:
        print("❌ 前端服务启动失败！")
        output, _ = process.communicate()
        print(output)
        return False
    
    print("✅ Streamlit前端已启动: http://localhost:8501")
    return True


def cleanup():
    """清理进程"""
    print("\n🧹 正在关闭所有服务...")
    
    for name, process in processes:
        if process.poll() is None:  # 进程仍在运行
            try:
                if os.name == "nt":
                    # Windows: 发送CTRL_BREAK_EVENT到进程组
                    os.kill(process.pid, signal.CTRL_BREAK_EVENT)
                else:
                    # Unix: 终止整个进程组
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                print(f"✅ {name} 已关闭")
            except Exception as e:
                print(f"⚠️ 关闭 {name} 时出错: {e}")
                # 强制终止
                try:
                    process.terminate()
                except:
                    pass


def main():
    """主函数"""
    print("\n" + "🏥 " + "=" * 44 + " 🏥")
    print("   医疗知识问答系统 - 一键启动")
    print("=" * 46)
    
    # 注册退出时的清理函数
    atexit.register(cleanup)
    
    # 注册信号处理
    def signal_handler(sig, frame):
        print("\n\n⚠️ 收到中断信号，正在关闭服务...")
        cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    if os.name != "nt":
        signal.signal(signal.SIGTERM, signal_handler)
    
    # 启动后端
    if not start_backend():
        print("\n❌ 后端启动失败，程序退出")
        sys.exit(1)
    
    # 启动前端
    if not start_frontend():
        print("\n❌ 前端启动失败，后端已启动，请手动关闭")
        sys.exit(1)
    
    # 打印完成信息
    print("\n" + "=" * 50)
    print("🎉 启动完成！")
    print("=" * 50)
    print("📱 访问地址:")
    print("   • 前端: http://localhost:8501")
    print("   • API文档: http://localhost:8000/docs")
    print("   • 健康检查: http://localhost:8000/health")
    print("\n⚠️ 按 Ctrl+C 可停止所有服务")
    print("=" * 50 + "\n")
    
    # 保持脚本运行，实时显示输出
    try:
        # 监控进程输出
        import select
        
        while True:
            has_output = False
            
            for name, process in processes:
                if process.poll() is None:
                    # 尝试读取输出（Windows下使用非阻塞读取）
                    try:
                        import msvcrt
                        while msvcrt.kbhit():
                            char = msvcrt.getch()
                            if char:
                                has_output = True
                    except ImportError:
                        # Unix系统
                        pass
            
            if not has_output:
                time.sleep(0.5)
                
            # 检查是否有进程退出
            for name, process in processes:
                if process.poll() is not None:
                    print(f"\n❌ {name} 进程已退出!")
                    cleanup()
                    sys.exit(1)
                    
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


if __name__ == "__main__":
    main()
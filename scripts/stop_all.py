"""
一键关闭医疗知识问答系统（后端API + 前端Streamlit）
"""

import os
import sys
import signal
import subprocess
import time


def find_process_by_port(port):
    """查找占用指定端口的进程"""
    try:
        # Windows: 使用 netstat 查找端口
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True
        )
        
        for line in result.stdout.split("\n"):
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    try:
                        return int(pid)
                    except ValueError:
                        pass
    except Exception as e:
        print(f"查找端口 {port} 进程时出错: {e}")
    return None


def kill_process(pid):
    """强制终止进程"""
    try:
        if os.name == "nt":
            # Windows: 使用 taskkill
            subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"], check=True)
        else:
            # Unix: 发送 SIGTERM
            os.kill(pid, signal.SIGTERM)
        return True
    except Exception as e:
        print(f"  终止进程 {pid} 失败: {e}")
        return False


def find_streamlit_processes():
    """查找所有Streamlit相关进程"""
    processes = []
    
    try:
        if os.name == "nt":
            # Windows: 使用 tasklist
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq python.exe"],
                capture_output=True,
                text=True
            )
            
            for line in result.stdout.split("\n"):
                if "python.exe" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        pid = int(parts[1])
                        # 获取命令行参数
                        try:
                            info = subprocess.run(
                                ["wmic", "process", "where", f"ProcessId={pid}", "get", "CommandLine"],
                                capture_output=True,
                                text=True
                            )
                            cmdline = info.stdout
                            if "streamlit" in cmdline.lower() or "uvicorn" in cmdline.lower():
                                processes.append(pid)
                        except:
                            pass
                            
        else:
            # Unix: 使用 ps
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True
            )
            
            for line in result.stdout.split("\n"):
                if "streamlit" in line or "uvicorn" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            pid = int(parts[1])
                            processes.append(pid)
                        except ValueError:
                            pass
                            
    except Exception as e:
        print(f"查找进程时出错: {e}")
    
    return processes


def main():
    """主函数"""
    print("\n" + "🛑 " + "=" * 44 + " 🛑")
    print("   医疗知识问答系统 - 一键关闭")
    print("=" * 46)
    
    killed_count = 0
    
    # 1. 查找并关闭占用 8000 端口的进程（后端API）
    print("\n📡 查找后端API服务 (端口 8000)...")
    backend_pid = find_process_by_port(8000)
    if backend_pid:
        print(f"   找到后端进程 PID: {backend_pid}")
        if kill_process(backend_pid):
            print("   ✅ 后端服务已关闭")
            killed_count += 1
        else:
            print("   ❌ 后端服务关闭失败")
    else:
        print("   ℹ️ 未找到运行中的后端服务")
    
    # 2. 查找并关闭占用 8501 端口的进程（Streamlit前端）
    print("\n🌐 查找Streamlit前端服务 (端口 8501)...")
    frontend_pid = find_process_by_port(8501)
    if frontend_pid:
        print(f"   找到前端进程 PID: {frontend_pid}")
        if kill_process(frontend_pid):
            print("   ✅ 前端服务已关闭")
            killed_count += 1
        else:
            print("   ❌ 前端服务关闭失败")
    else:
        print("   ℹ️ 未找到运行中的前端服务")
    
    # 3. 额外检查：查找所有相关的 Python 进程（streamlit/uvicorn）
    print("\n🔍 额外检查相关进程...")
    related_processes = find_streamlit_processes()
    if related_processes:
        print(f"   找到 {len(related_processes)} 个相关进程:")
        for pid in related_processes:
            print(f"     - PID: {pid}")
            if kill_process(pid):
                print(f"       ✅ 已终止")
                killed_count += 1
            else:
                print(f"       ❌ 终止失败")
    else:
        print("   ℹ️ 未找到其他相关进程")
    
    # 完成
    print("\n" + "=" * 50)
    if killed_count > 0:
        print(f"✅ 成功关闭 {killed_count} 个进程")
    else:
        print("ℹ️ 没有需要关闭的服务")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()

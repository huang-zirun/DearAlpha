#!/usr/bin/env python3
"""
DearAlpha 交互式启动器
======================

提供友好的命令行菜单，让用户交互式选择挖掘模式。

Usage:
  python start.py
"""

import sys
import subprocess
from pathlib import Path


def print_banner():
    """打印欢迎横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██████╗ ███████╗ █████╗ ██████╗      █████╗ ██╗     ██╗   ║
║   ██╔══██╗██╔════╝██╔══██╗██╔══██╗    ██╔══██╗██║     ██║   ║
║   ██║  ██║█████╗  ███████║██████╔╝    ███████║██║     ██║   ║
║   ██║  ██║██╔══╝  ██╔══██║██╔══██╗    ██╔══██║██║     ██║   ║
║   ██████╔╝███████╗██║  ██║██║  ██║    ██║  ██║███████╗██║   ║
║   ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚══════╝╚═╝   ║
║                                                              ║
║              WorldQuant Brain Alpha 因子挖掘框架              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_menu():
    """打印主菜单"""
    menu = """
┌─────────────────────────────────────────────────────────────┐
│  请选择挖掘模式:                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [1] 🤖 AI 模式        - LLM 生成裸信号（发现式）           │
│  [2] 🔄 Pipeline 模式  - 三阶递进流水线（day1→day2→day3）   │
│  [3] 📊 Template 模式  - 笛卡尔积穷举内置模板库             │
│  [4] 🎯 Layered 模式   - 分层剪枝（大字段空间推荐）         │
│  [5] 🧠 Bayesian 模式  - 贝叶斯优化数值参数                 │
│  [6] 📤 Submit 模式    - 提交通过质量门控的 alpha           │
│                                                             │
│  [0] ❌ 退出                                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
"""
    print(menu)


def get_choice():
    """获取用户选择"""
    while True:
        choice = input("请输入选项 [0-6]: ").strip()
        if choice in ["0", "1", "2", "3", "4", "5", "6"]:
            return choice
        print("❌ 无效选项，请重新输入")


def run_ai_mode():
    """运行 AI 模式"""
    print("\n" + "=" * 50)
    print("🤖 AI 模式配置")
    print("=" * 50)
    
    rounds = input("请输入轮数 (默认: 1): ").strip() or "1"
    theme = input("请输入主题 (可选，如 Momentum/Value，直接回车跳过): ").strip()
    verbose = input("是否开启详细日志? [y/N]: ").strip().lower() == "y"
    
    cmd = ["python", "mine.py", "ai", "--rounds", rounds]
    if theme:
        cmd.extend(["--theme", theme])
    if verbose:
        cmd.append("--verbose")
    
    return cmd


def run_pipeline_mode():
    """运行 Pipeline 模式"""
    print("\n" + "=" * 50)
    print("🔄 Pipeline 模式配置")
    print("=" * 50)
    
    fields = input("请输入字段 (可选，逗号分隔，如 close,volume,returns，直接回车使用配置): ").strip()
    field_prefix = input("请输入字段前缀 (可选，如 anl4，直接回车跳过): ").strip()
    prune_keep = input("每字段保留数量 (可选，默认 5): ").strip() or ""
    verbose = input("是否开启详细日志? [y/N]: ").strip().lower() == "y"
    
    cmd = ["python", "mine.py", "pipeline"]
    if fields:
        cmd.extend(["--fields", fields])
    if field_prefix:
        cmd.extend(["--field-prefix", field_prefix])
    if prune_keep:
        cmd.extend(["--prune-keep", prune_keep])
    if verbose:
        cmd.append("--verbose")
    
    return cmd


def run_template_mode():
    """运行 Template 模式"""
    print("\n" + "=" * 50)
    print("📊 Template 模式配置")
    print("=" * 50)
    
    verbose = input("是否开启详细日志? [y/N]: ").strip().lower() == "y"
    
    cmd = ["python", "mine.py", "template"]
    if verbose:
        cmd.append("--verbose")
    
    return cmd


def run_layered_mode():
    """运行 Layered 模式"""
    print("\n" + "=" * 50)
    print("🎯 Layered 模式配置")
    print("=" * 50)
    
    template = input("请输入模板 (可选，直接回车使用默认): ").strip()
    fields = input("请输入字段 (可选，逗号分隔，直接回车使用默认): ").strip()
    keep_fields = input("保留字段数量 (可选，默认 5): ").strip() or ""
    coarse_windows = input("粗筛窗口 (可选，如 5,22,120，直接回车使用默认): ").strip()
    verbose = input("是否开启详细日志? [y/N]: ").strip().lower() == "y"
    
    cmd = ["python", "mine.py", "layered"]
    if template:
        cmd.extend(["--template", template])
    if fields:
        cmd.extend(["--fields", fields])
    if keep_fields:
        cmd.extend(["--keep-fields", keep_fields])
    if coarse_windows:
        cmd.extend(["--coarse-windows", coarse_windows])
    if verbose:
        cmd.append("--verbose")
    
    return cmd


def run_bayesian_mode():
    """运行 Bayesian 模式"""
    print("\n" + "=" * 50)
    print("🧠 Bayesian 模式配置")
    print("=" * 50)
    
    template = input("请输入模板 (可选，直接回车使用默认): ").strip()
    n_trials = input("试验次数 (可选，默认 50): ").strip() or ""
    verbose = input("是否开启详细日志? [y/N]: ").strip().lower() == "y"
    
    cmd = ["python", "mine.py", "bayesian"]
    if template:
        cmd.extend(["--template", template])
    if n_trials:
        cmd.extend(["--n-trials", n_trials])
    if verbose:
        cmd.append("--verbose")
    
    return cmd


def run_submit_mode():
    """运行 Submit 模式"""
    print("\n" + "=" * 50)
    print("📤 Submit 模式配置")
    print("=" * 50)
    
    dry_run = input("是否仅预览 (dry-run)? [Y/n]: ").strip().lower() != "n"
    verbose = input("是否开启详细日志? [y/N]: ").strip().lower() == "y"
    
    cmd = ["python", "mine.py", "submit"]
    if dry_run:
        cmd.append("--dry-run")
    if verbose:
        cmd.append("--verbose")
    
    return cmd


def confirm_and_run(cmd):
    """确认并执行命令"""
    print("\n" + "=" * 50)
    print("📋 即将执行命令:")
    print("   " + " ".join(cmd))
    print("=" * 50)
    
    confirm = input("\n确认执行? [Y/n]: ").strip().lower()
    if confirm == "n":
        print("❌ 已取消")
        return False
    
    print("\n" + "=" * 50)
    print("🚀 启动挖掘...")
    print("=" * 50 + "\n")
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 执行失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
        return False


def main():
    """主函数"""
    print_banner()
    
    # 检查 mine.py 是否存在
    if not Path("mine.py").exists():
        print("❌ 错误: 未找到 mine.py，请确保在 DearAlpha 项目根目录运行")
        sys.exit(1)
    
    while True:
        print_menu()
        choice = get_choice()
        
        if choice == "0":
            print("\n👋 再见!")
            break
        
        # 根据选择构建命令
        if choice == "1":
            cmd = run_ai_mode()
        elif choice == "2":
            cmd = run_pipeline_mode()
        elif choice == "3":
            cmd = run_template_mode()
        elif choice == "4":
            cmd = run_layered_mode()
        elif choice == "5":
            cmd = run_bayesian_mode()
        elif choice == "6":
            cmd = run_submit_mode()
        else:
            continue
        
        # 执行命令
        confirm_and_run(cmd)
        
        # 询问是否继续
        print("\n" + "=" * 50)
        cont = input("是否返回主菜单? [Y/n]: ").strip().lower()
        if cont == "n":
            print("\n👋 再见!")
            break
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 再见!")
        sys.exit(0)

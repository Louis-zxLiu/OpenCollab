#!/usr/bin/env python3
"""
OpenCollab 演示脚本
展示多智能体协作的基本用法
"""

import asyncio
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from opencollab.team.orchestrator import Team


async def demo_simple_task():
    """演示简单任务委托"""
    print("=" * 60)
    print("演示 1: 简单任务委托")
    print("=" * 60)
    
    # 创建团队
    team = Team(
        workspace=".",
        model="gpt-4o",
        provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        use_worktrees=False  # 演示使用本地环境
    )
    
    try:
        # 委托一个简单任务
        result = await team.delegate(
            "analyst",
            "分析这个项目的结构，总结主要文件和目录的作用。"
        )
        print(f"\n分析结果:\n{result[:500]}...")
    except Exception as e:
        print(f"错误: {e}")
    finally:
        await team.cleanup()


async def demo_self_collaboration():
    """演示自协作（Coder + Reviewer）"""
    print("\n" + "=" * 60)
    print("演示 2: 自协作（代码实现 + 审查）")
    print("=" * 60)
    
    team = Team(
        workspace=".",
        model="gpt-4o",
        provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        use_worktrees=False
    )
    
    try:
        # 使用delegate_with_review进行自协作
        result = await team.delegate_with_review(
            task="编写一个Python函数，计算斐波那契数列的第n项，要求使用递归实现。"
        )
        print(f"\n自协作结果:\n{result[:500]}...")
    except Exception as e:
        print(f"错误: {e}")
    finally:
        await team.cleanup()


async def demo_team_workflow():
    """演示完整团队工作流"""
    print("\n" + "=" * 60)
    print("演示 3: 完整团队工作流（Lead协调多个角色）")
    print("=" * 60)
    
    team = Team(
        workspace=".",
        model="gpt-4o",
        provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        use_worktrees=False
    )
    
    try:
        # 用户向Lead提出请求
        user_request = """
        我需要为项目添加一个简单的命令行工具，功能如下：
        1. 接收一个目录路径作为参数
        2. 统计该目录下所有Python文件的总行数
        3. 输出统计结果
        
        请协调团队完成这个任务。
        """
        
        result = await team.run(user_request)
        print(f"\n团队工作流结果:\n{result[:800]}...")
    except Exception as e:
        print(f"错误: {e}")
    finally:
        await team.cleanup()


async def demo_parallel_tasks():
    """演示并行任务"""
    print("\n" + "=" * 60)
    print("演示 4: 并行任务执行")
    print("=" * 60)
    
    team = Team(
        workspace=".",
        model="gpt-4o",
        provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        use_worktrees=True  # 使用worktree实现物理隔离
    )
    
    try:
        # 创建多个并行任务
        tasks = [
            team.delegate("analyst", "分析README.md文件的内容"),
            team.delegate("coder", "查看core目录下的主要文件"),
            team.delegate("reviewer", "检查项目结构是否符合最佳实践")
        ]
        
        # 并行执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (role, result) in enumerate(zip(["analyst", "coder", "reviewer"], results)):
            if isinstance(result, Exception):
                print(f"\n{role} 任务失败: {result}")
            else:
                print(f"\n{role} 结果:\n{str(result)[:300]}...")
    except Exception as e:
        print(f"错误: {e}")
    finally:
        await team.cleanup()


async def main():
    """主函数"""
    print("OpenCollab 功能演示")
    print("=" * 60)
    
    # 检查API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("警告: 未设置OPENAI_API_KEY环境变量")
        print("请在.env文件中设置或直接导出环境变量")
        print("演示将使用模拟模式（如果实现）\n")
    
    # 运行演示
    demos = [
        ("简单任务委托", demo_simple_task),
        ("自协作", demo_self_collaboration),
        ("团队工作流", demo_team_workflow),
        ("并行任务", demo_parallel_tasks),
    ]
    
    for name, demo_func in demos:
        try:
            await demo_func()
        except Exception as e:
            print(f"\n{name} 演示失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

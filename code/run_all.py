#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键运行 hw05 所有任务
包含：任务一 (极简CNN) + 任务二 (LeNet-5)
"""

import subprocess
import sys
import os

def run_script(script_name):
    """
    运行指定的Python脚本
    :param script_name: 脚本文件名 (如 'minimal_cnn.py')
    """
    print(f"\n{'='*50}")
    print(f"正在开始运行: {script_name}")
    print(f"{'='*50}")
    
    # 使用 sys.executable 保证调用当前环境的Python解释器
    # cwd 设置为脚本所在目录，防止路径问题
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    try:
        # check=True 会在脚本报错时自动抛出异常
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(script_path),
            check=True,
            text=True,
            encoding='utf-8'
        )
        print(f"\n✅ {script_name} 运行完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 【错误】{script_name} 运行失败！")
        print(f"错误输出: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"错误：未找到文件 {script_path}")
        return False

if __name__ == "__main__":
    print("开始执行 HW05 自动化训练流程...")
    
    # 1. 运行任务一：极简 CNN
    success1 = run_script("minimal_cnn.py")
    
    # 2. 运行任务二：LeNet-5
    success2 = run_script("train_lenet.py")
    
    # 最终总结
    print(f"\n{'='*60}")
    print("流程总结：")
    print(f"极简CNN: {'✅ 成功' if success1 else '❌ 失败'}")
    print(f"LeNet-5: {'✅ 成功' if success2 else '❌ 失败'}")
    print(f"{'='*60}")
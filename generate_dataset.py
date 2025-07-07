# -*- coding: utf-8 -*-
"""
文件名: generate_dataset.py
描述: 调用仿真引擎，批量生成用于神经网络训练的数据集。
      本脚本模拟一个【物理弱化】的电网，并只对【计算收敛】的样本进行分析。
"""
import pandas as pd
from scipy.stats import qmc
from pathlib import Path
import os
from simulation_engine import SimulationEngine
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 结果分析与绘图函数 ---
def analyze_and_plot_results(df, results_dir):
    """
    分析【已收敛】的数据集，统计PV渗透率对电压安全的影响，并生成可视化散点图。
    """
    print("\n\n--- 结果分析与可视化 (仅限计算收敛的样本) ---")
    
    if df.empty:
        print("⚠️ 数据集为空或所有样本均为计算不收敛，无法进行电压安全分析。")
        return

    # 1. 整体统计分析
    safe_samples = df[df['label'] == 0]
    unsafe_samples = df[df['label'] == 1]

    if unsafe_samples.empty:
        print("✅ 在所有计算收敛的样本中，未发现任何电压越限问题。")
        return

    avg_pv_safe = safe_samples['pv_penetration'].mean()
    avg_pv_unsafe = unsafe_samples['pv_penetration'].mean()
    
    print(f"📊 整体统计摘要:")
    print(f"  - 安全样本的平均PV渗透率: {avg_pv_safe:.2f}")
    print(f"  - 电压越限样本的平均PV渗透率: {avg_pv_unsafe:.2f}")

    # 2. 按PV渗透率区间进行详细统计
    print("\n--- 按PV渗透率区间的电压安全影响分析 ---")
    bins = [0.5, 2.0, 4.0, 6.0]
    labels = ['0.5-2.0 (中)', '2.0-4.0 (高)', '4.0-6.0 (非常高)']
    df['pv_range'] = pd.cut(df['pv_penetration'], bins=bins, labels=labels, right=False, include_lowest=True)

    range_analysis = df.groupby('pv_range', observed=True).agg(
        total_samples=('label', 'count'),
        unsafe_samples=('label', 'sum')
    ).reset_index()

    range_analysis['unsafe_rate_%'] = np.where(
        range_analysis['total_samples'] > 0,
        (range_analysis['unsafe_samples'] / range_analysis['total_samples'] * 100),
        0
    ).round(2)

    print("PV渗透率区间对电压安全状态的影响:")
    print(range_analysis.to_string(index=False))
    print("\n  - 分析显示，随着PV渗透率区间的提高，电压越限的不安全率应呈现显著上升趋势。")

    # 3. 生成散点图
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"无法设置中文字体'SimHei'，绘图可能出现乱码: {e}")

    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {0: 'green', 1: 'red'}
    legend_labels = {0: '电压安全 (Safe)', 1: '电压越限 (Unsafe)'}

    sns.scatterplot(data=df, x='pv_penetration', y='load_multiplier', hue='label', 
                    palette=colors, style='label', s=80, ax=ax)
    
    ax.set_title('电网电压安全边界 (物理弱化模型)\n(Voltage Security Boundary with Physically Weakened Grid)', fontsize=16)
    ax.set_xlabel('光伏渗透率 (PV Penetration)', fontsize=12)
    ax.set_ylabel('负荷乘数 (Load Multiplier)', fontsize=12)
    
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, [legend_labels[0], legend_labels[1]], title='场景标签 (Label)')

    plot_path = results_dir / "safety_boundary_final.png"
    fig.savefig(plot_path, dpi=300)
    
    print(f"\n📈 可视化散点图已生成并保存到: '{plot_path}'")

# --- 1. 定义采样空间和常量 ---
NUM_SAMPLES = 500
V_MIN = 0.95
V_MAX = 1.05

parameter_bounds = {
    'pv_penetration': [0.5, 6.0],
    'load_multiplier': [0.5, 1.2]
}

# --- 2. 使用LHS生成输入样本 ---
l_bounds = [b[0] for b in parameter_bounds.values()]
u_bounds = [b[1] for b in parameter_bounds.values()]

sampler = qmc.LatinHypercube(d=len(parameter_bounds))
sample_points = qmc.scale(sampler.random(n=NUM_SAMPLES), l_bounds, u_bounds)
print(f"✅ 已使用LHS在2维空间中生成 {NUM_SAMPLES} 个采样点。")
print(f"   采样范围: PV渗透率 [{parameter_bounds['pv_penetration'][0]}, {parameter_bounds['pv_penetration'][1]}], 负荷乘数 [{parameter_bounds['load_multiplier'][0]}, {parameter_bounds['load_multiplier'][1]}]")


# --- 3. 初始化仿真引擎 ---
script_dir = Path(__file__).parent.resolve()
dss_file = script_dir / "IEEE13_Master.dss"
results_dir = script_dir / "results"
engine = SimulationEngine(str(dss_file), str(results_dir))

# --- 4. 循环运行仿真并生成数据集 ---
dataset = []
total_load_kw = 3465 
unsafe_count = 0

print("\n--- 开始仿真循环 (物理弱化电网 + 稳健求解器) ---")
for i, point in enumerate(sample_points):
    pv_penetration = point[0]
    load_multiplier = point[1]

    print(f"\r正在仿真场景 {i+1}/{NUM_SAMPLES}: PV Pen={pv_penetration:.2f}, Load Mult={load_multiplier:.2f}...", end="")

    total_pv_pmpp = total_load_kw * pv_penetration
    
    scenario_config = {
        'load_multiplier': load_multiplier,
        'pv_systems': {
            'PV_675': {'phases': 3, 'bus': '675', 'pmpp': total_pv_pmpp * 0.6, 'kVA': total_pv_pmpp * 0.6 * 1.1},
            'PV_611': {'phases': 1, 'bus': '611', 'pmpp': total_pv_pmpp * 0.4, 'kVA': total_pv_pmpp * 0.4 * 1.1}
        }
    }

    voltage_profiles, converged, reason = engine.run_24h_simulation(scenario_config)

    # --- 5. 标记场景 ---
    label = 0
    is_unsafe = False
    
    if not converged:
        label = -1 
    elif not voltage_profiles:
        is_unsafe = True
    else:
        for bus, df_v in voltage_profiles.items():
            engine.dss_engine.ActiveCircuit.SetActiveBus(bus)
            kv_base = engine.dss_engine.ActiveCircuit.ActiveBus.kVBase
            if kv_base == 0: continue
            
            phase_base_voltage = (kv_base * 1000.0) / math.sqrt(3.0) if engine.dss_engine.ActiveCircuit.ActiveBus.NumNodes >= 3 else (kv_base * 1000.0)
            
            for col in df_v.columns:
                if col.startswith('V') and 'Angle' not in col:
                    if phase_base_voltage == 0: continue
                    v_pu_series = df_v[col] / phase_base_voltage
                    if not v_pu_series[(v_pu_series >= V_MIN) & (v_pu_series <= V_MAX)].all():
                        is_unsafe = True
                        break
            if is_unsafe:
                break
    
    if is_unsafe:
        label = 1
        unsafe_count += 1

    dataset.append({
        'pv_penetration': pv_penetration,
        'load_multiplier': load_multiplier,
        'label': label,
        'unsafe_reason': reason 
    })

# --- 6. 保存并统计数据集 ---
df = pd.DataFrame(dataset)
csv_path = results_dir / "dataset_final_ultimate.csv"
df.to_csv(csv_path, index=False)

print(f"\n\n🎉 数据集生成完毕！ {NUM_SAMPLES} 个场景已仿真并保存到 '{csv_path}'。")
print(f"   总共发现 {unsafe_count} 个电压越限样本。")
print(f"   另有 {(df['label'] == -1).sum()} 个计算不收敛样本。")

# --- 7. 创建一个只包含电压问题的“干净”数据集 ---
df_clean = df[df['label'] != -1].copy()

if not df_clean.empty:
    clean_csv_path = results_dir / "dataset_clean_voltage_only_ultimate.csv"
    df_clean.to_csv(clean_csv_path, index=False)
    print(f"\n\n✅ 已创建只包含【计算收敛】样本的干净数据集，并保存到 '{clean_csv_path}'。")
    print("\n--- 干净数据集统计 (安全 vs 电压越限) ---")
    print(df_clean['label'].value_counts())
else:
    print("\n\n⚠️ 未能生成任何计算收敛的样本。")

# --- 8. 调用分析与绘图功能，并传入【干净】的数据集 ---
analyze_and_plot_results(df_clean, results_dir)

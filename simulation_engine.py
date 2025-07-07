# -*- coding: utf-8 -*-
"""
文件名: simulation_engine.py
描述: 一个可重用的、基于最佳实践的OpenDSS时序仿真引擎。
"""
import dss
import pandas as pd
import os
from pathlib import Path
import math

class SimulationEngine:
    def __init__(self, master_dss_file, results_dir):
        self.dss_engine = dss.DSS
        self.master_dss_file = master_dss_file
        self.results_dir = Path(results_dir)
        os.makedirs(self.results_dir, exist_ok=True)

    def _compile_master_file(self):
        """加载基础电网拓扑。"""
        self.dss_engine.Text.Command = "Clear"
        self.dss_engine.Text.Command = f"Compile [{self.master_dss_file}]"
        if "ieee13node" not in self.dss_engine.ActiveCircuit.Name.lower():
            raise Exception("基础模型加载失败或名称不匹配。")

    def _define_scenario_components(self, scenario_config):
        """根据场景配置，以编程方式定义所有动态和可变元件。"""
        pv_config = scenario_config.get('pv_systems', {})
        load_multiplier = scenario_config.get('load_multiplier', 1.0)

        # 定义负荷
        self.dss_engine.Text.Command = f"New Load.671 phases=3 bus1=671 conn=wye model=5 kv=4.16 kw={1155*load_multiplier} kvar={660*load_multiplier}"
        self.dss_engine.Text.Command = f"New Load.645 phases=2 bus1=645.1.2 conn=wye model=5 kv=2.4 kw={170*load_multiplier} kvar={125*load_multiplier}"
        self.dss_engine.Text.Command = f"New Load.646 phases=2 bus1=646.1.2 conn=wye model=5 kv=2.4 kw={230*load_multiplier} kvar={132*load_multiplier}"
        self.dss_engine.Text.Command = f"New Load.692 phases=3 bus1=692.1.2.3 conn=delta model=5 kv=4.16 kw={170*load_multiplier} kvar={151*load_multiplier}"
        self.dss_engine.Text.Command = f"New Load.675 phases=3 bus1=675.1.2.3 conn=wye model=5 kv=4.16 kw={485*load_multiplier} kvar={190*load_multiplier}"
        self.dss_engine.Text.Command = f"New Load.611 phases=1 bus1=611.3 conn=wye model=5 kv=2.4 kw={170*load_multiplier} kvar={80*load_multiplier}"
        self.dss_engine.Text.Command = f"New Load.652 phases=1 bus1=652.1 conn=wye model=5 kv=2.4 kw={128*load_multiplier} kvar={86*load_multiplier}"

        # 启用电容器
        self.dss_engine.Text.Command = "New Capacitor.Cap1 phases=3 bus1=675 conn=wye kv=4.16 kvar=600"
        self.dss_engine.Text.Command = "New Capacitor.Cap2 phases=1 bus1=611.3 conn=wye kv=2.4 kvar=100"
        
        # 恢复调节器控制，并设定一个适中的带宽(band=3)
        self.dss_engine.Text.Command = "New RegControl.Reg1 transformer=Reg1 winding=2 vreg=123.6 band=3"
        self.dss_engine.Text.Command = "New RegControl.Reg2 transformer=Reg2 winding=2 vreg=123.6 band=3"
        self.dss_engine.Text.Command = "New RegControl.Reg3 transformer=Reg3 winding=2 vreg=123.6 band=3"

        # 定义并关联负荷曲线
        self.dss_engine.Text.Command = "New LoadShape.Residential npts=24 interval=1 mult=(0.3 0.3 0.3 0.4 0.5 0.6 0.7 0.8 0.8 0.8 0.8 0.8 0.9 0.9 0.9 1.0 1.1 1.2 1.1 1.0 0.9 0.8 0.6 0.4)"
        load_idx = self.dss_engine.ActiveCircuit.Loads.First
        while load_idx > 0:
            self.dss_engine.ActiveCircuit.Loads.daily = "Residential"
            load_idx = self.dss_engine.ActiveCircuit.Loads.Next
        
        # 定义一个峰值更高的PV出力曲线，以模拟极端天气
        self.dss_engine.Text.Command = "New LoadShape.PV_Shape npts=24 interval=1 mult=(0 0 0 0 0 0 0.1 0.4 0.8 1.0 1.1 1.2 1.1 1.0 0.8 0.5 0.2 0.1 0 0 0 0 0 0)"
        
        # 根据配置添加光伏系统
        for pv_name, pv_details in pv_config.items():
            self.dss_engine.Text.Command = f"New PVSystem.{pv_name} phases={pv_details['phases']} bus1={pv_details['bus']} conn=wye kVA={pv_details['kVA']} pmpp={pv_details['pmpp']} pf=1.0 daily=PV_Shape"

    def run_24h_simulation(self, scenario_config):
        self._compile_master_file()
        self._define_scenario_components(scenario_config)

        load_buses_to_monitor = ["671", "645", "646", "692", "675", "611", "652"]
        for bus in load_buses_to_monitor:
            element_name = f"Load.{bus}"
            self.dss_engine.Text.Command = f"New Monitor.{bus}_V element={element_name} terminal=1 mode=0"

        # 【核心修改】: 更换为更稳健的潮流算法，以确保在极端情况下也能收敛
        self.dss_engine.Text.Command = "Set Algorithm=Normal"
        self.dss_engine.Text.Command = "Set mode=daily number=24 stepsize=1h"
        self.dss_engine.Text.Command = "Solve"

        if not self.dss_engine.ActiveCircuit.Solution.Converged:
            return None, False, 'NonConvergence'

        self.dss_engine.Text.Command = f"Set Datapath={self.results_dir}"
        voltage_profiles = {}
        for bus in load_buses_to_monitor:
            monitor_name = f"{bus}_V"
            self.dss_engine.Text.Command = f"Export Monitors {monitor_name}"
            
            circuit_name = self.dss_engine.ActiveCircuit.Name
            csv_path = self.results_dir / f"{circuit_name}_Mon_{monitor_name}_1.csv"
            
            if not csv_path.exists(): continue

            df_v = pd.read_csv(csv_path)
            df_v.columns = [col.strip() for col in df_v.columns]
            voltage_profiles[bus] = df_v

        return voltage_profiles, True, 'Converged'

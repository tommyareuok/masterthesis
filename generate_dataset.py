# -*- coding: utf-8 -*-
"""
This script generates a refined dataset for thesis research using the mv_oberrhein
network. It employs a "boundary discovery" strategy by sampling uniformly across
a wide operational space and labeling points based on simulation outcomes.

Correction: This version fixes a TypeError by implementing a two-snapshot
simulation (peak PV and peak EV) for each sample to ensure robust stability checks,
rather than attempting a single, ambiguous time-series run.
"""
import pandapower as pp
import pandapower.networks as nw
import numpy as np
import pandas as pd
from scipy.stats import qmc
from tqdm import tqdm
from copy import deepcopy
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Simulation Constants ---
TIMESTEPS = 24
VM_MIN_PU = 0.95
VM_MAX_PU = 1.05
MAX_LINE_LOADING = 70.0
EV_CHARGER_KW = 22.0
PV_POWER_FACTOR = 0.95

# --- Unified Sampling Space ---
TOTAL_SAMPLES = 200
PV_PENETRATION_RANGE = [0.1, 2.5]
EV_COUNT_RANGE = [0, 200]

def get_grid():
    """Initializes and returns the pandapower medium voltage oberrhein network."""
    net = nw.mv_oberrhein()
    net.load.drop(net.load.index, inplace=True)
    net.sgen.drop(net.sgen.index, inplace=True)
    net.gen.drop(net.gen.index, inplace=True)
    return net

def create_profiles():
    """Creates normalized 24-hour load, PV, and EV charging profiles."""
    profiles = {
        'load': np.array([0.6, 0.5, 0.4, 0.4, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.8, 0.9,
                          0.9, 0.8, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6]),
        'pv': np.array([0, 0, 0, 0, 0, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0,
                        0.9, 0.7, 0.5, 0.3, 0.1, 0, 0, 0, 0, 0, 0, 0]),
        'ev': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0.1, 0.4, 0.8, 1.0, 1.0, 0.8, 0.5, 0.2])
    }
    return profiles

def run_simulation_for_sample(base_net, profiles, pv_penetration, ev_count):
    """
    Runs a two-snapshot simulation for a single sample (peak PV and peak EV).
    The sample is safe only if it passes checks at both critical timesteps.
    """
    total_load_p_mw = 17.285  # From original network description
    pv_p_mw_base = total_load_p_mw * pv_penetration

    # Define asset locations once per sample for consistency
    pv_buses = [1, 2, 6, 8, 10, 12, 14, 16]
    ev_buses = np.random.choice(base_net.bus.index, size=int(ev_count), replace=True)
    base_load_buses = [1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 17]

    # --- Snapshot 1: Peak PV (Timestep 11) ---
    ts_pv_peak = 11
    net_pv = deepcopy(base_net)
    
    # Create PVs at peak generation
    for bus_idx in pv_buses:
        p_mw = (pv_p_mw_base / len(pv_buses)) * profiles['pv'][ts_pv_peak]
        q_mvar = -p_mw * np.tan(np.arccos(PV_POWER_FACTOR)) if p_mw > 0 else 0
        pp.create_sgen(net_pv, bus=bus_idx, p_mw=p_mw, q_mvar=q_mvar, type='pv')

    # Create EVs and loads at corresponding time
    for bus_idx in ev_buses:
        pp.create_load(net_pv, bus=bus_idx, p_mw=(EV_CHARGER_KW / 1000) * profiles['ev'][ts_pv_peak], type='ev_charger')
    for bus_idx in base_load_buses:
        pp.create_load(net_pv, bus=bus_idx, p_mw=(1.0 / len(base_load_buses)) * profiles['load'][ts_pv_peak],
                       q_mvar=(0.5 / len(base_load_buses)) * profiles['load'][ts_pv_peak], type='base')

    try:
        pp.runpp(net_pv)
        if net_pv.res_bus.vm_pu.max() > VM_MAX_PU or net_pv.res_bus.vm_pu.min() < VM_MIN_PU:
            return {'label': 1, 'label_reason': 'unsafe_voltage_pv_peak'}
        if net_pv.res_line.loading_percent.max() > MAX_LINE_LOADING:
            return {'label': 1, 'label_reason': 'unsafe_loading_pv_peak'}
    except pp.LoadflowNotConverged:
        return {'label': 1, 'label_reason': 'unconverged_pv_peak'}

    # --- Snapshot 2: Peak EV (Timestep 19) ---
    ts_ev_peak = 19
    net_ev = deepcopy(base_net)

    # Create PVs (will be zero at this time)
    for bus_idx in pv_buses:
        p_mw = (pv_p_mw_base / len(pv_buses)) * profiles['pv'][ts_ev_peak]
        pp.create_sgen(net_ev, bus=bus_idx, p_mw=p_mw, q_mvar=0, type='pv')

    # Create EVs at peak charging and corresponding loads
    for bus_idx in ev_buses:
        pp.create_load(net_ev, bus=bus_idx, p_mw=(EV_CHARGER_KW / 1000) * profiles['ev'][ts_ev_peak], type='ev_charger')
    for bus_idx in base_load_buses:
        pp.create_load(net_ev, bus=bus_idx, p_mw=(1.0 / len(base_load_buses)) * profiles['load'][ts_ev_peak],
                       q_mvar=(0.5 / len(base_load_buses)) * profiles['load'][ts_ev_peak], type='base')

    try:
        pp.runpp(net_ev)
        if net_ev.res_bus.vm_pu.max() > VM_MAX_PU or net_ev.res_bus.vm_pu.min() < VM_MIN_PU:
            return {'label': 1, 'label_reason': 'unsafe_voltage_ev_peak'}
        if net_ev.res_line.loading_percent.max() > MAX_LINE_LOADING:
            return {'label': 1, 'label_reason': 'unsafe_loading_ev_peak'}
    except pp.LoadflowNotConverged:
        return {'label': 1, 'label_reason': 'unconverged_ev_peak'}

    # If both snapshots are safe
    return {'label': 0, 'label_reason': 'safe'}

def generate_boundary_dataset(output_path='dataset_boundary.csv'):
    """Generates the dataset using the boundary discovery method."""
    print("--- Starting Dataset Generation for Boundary Discovery ---")
    
    print("1. Initializing grid and profiles...")
    net = get_grid()
    profiles = create_profiles()

    print(f"\n2. Setting up Latin Hypercube Sampler for {TOTAL_SAMPLES} samples...")
    sampler = qmc.LatinHypercube(d=2)
    sample_points = sampler.random(n=TOTAL_SAMPLES)
    scaled_points = qmc.scale(sample_points, 
                              [PV_PENETRATION_RANGE[0], EV_COUNT_RANGE[0]], 
                              [PV_PENETRATION_RANGE[1], EV_COUNT_RANGE[1]])
    
    print("\n3. Running simulations for all samples...")
    results = []
    for point in tqdm(scaled_points, desc="Simulating Scenarios"):
        pv_penetration, ev_count = point[0], int(point[1])
        sim_result = run_simulation_for_sample(net, profiles, pv_penetration, ev_count)
        results.append({
            'pv_penetration': pv_penetration, 'ev_count': ev_count,
            'label_reason': sim_result['label_reason'], 'label': sim_result['label']
        })
    
    final_df = pd.DataFrame(results)
    print(f"\n4. Simulation complete. Total samples: {len(final_df)}")
    final_df.to_csv(output_path, index=False)
    print(f"5. Dataset saved to {output_path}")
    
    safe_count = (final_df.label == 0).sum()
    unsafe_count = (final_df.label == 1).sum()
    print(f"   - Safe (0): {safe_count}, Unsafe (1): {unsafe_count}")
    return final_df

def plot_security_boundary(df, output_image_path='security_boundary_refined.png'):
    """Generates and saves a scatter plot visualizing the security boundary."""
    print(f"\n6. Generating security boundary plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    palette = {0: '#3498db', 1: '#e74c3c'}
    
    sns.scatterplot(data=df, x='pv_penetration', y='ev_count', hue='label',
                    palette=palette, s=20, alpha=0.7, ax=ax)
    
    ax.set_title('Security Boundary Visualization', fontsize=16, fontweight='bold')
    ax.set_xlabel('PV Penetration (p.u. of total load)', fontsize=12)
    ax.set_ylabel('Number of EV Chargers', fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['Safe (Label=0)', 'Unsafe (Label=1)'], title='Status', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    print(f"   - Plot saved to {output_image_path}")
    plt.show()

if __name__ == '__main__':
    dataset = generate_boundary_dataset()
    plot_security_boundary(dataset)
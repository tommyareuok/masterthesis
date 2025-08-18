# -*- coding: utf-8 -*-
"""
This script generates the final dataset for the thesis research. It uses the
robust mv_oberrhein network and simulates a more challenging scenario where
EV charging occurs during the day, overlapping with peak PV generation.

**REVISION**: This version implements a "combination punch" strategy:
1. EV charger power is increased to 22kW to enhance their impact.
2. PV inverters are modeled with a power factor of 0.95 (absorbing reactive
   power) to mitigate their overvoltage effect, creating a more balanced
   and complex interaction between the two uncertainty sources.
"""
import pandapower as pp
import pandapower.networks as nw
import numpy as np
import pandas as pd
from scipy.stats import qmc
from tqdm import tqdm
from copy import deepcopy
import os

# --- Simulation Constants ---
NUM_SAMPLES = 100
TIMESTEPS = 24
VM_MIN_PU = 0.95
VM_MAX_PU = 1.05
MAX_LINE_LOADING = 100.0

# --- Uncertainty Ranges ---
PV_PENETRATION_RANGE = [0.5, 1.8]
EV_COUNT_RANGE = [50, 350]
# **FIXED**: Increased single EV charger power to 22kW
EV_CHARGER_KW = 22.0
# **ADDED**: Power factor for PV inverters to simulate voltage support
PV_POWER_FACTOR = 0.95


def get_grid():
    """
    Loads the mv_oberrhein network.
    """
    net = nw.mv_oberrhein()
    return net

def create_profiles():
    """
    Creates normalized 24h load, PV, and EV profiles.
    """
    load_profile = np.array([
        0.60, 0.55, 0.52, 0.50, 0.52, 0.55, 0.65, 0.75, 0.85, 0.90, 0.95, 1.00,
        0.95, 0.90, 0.88, 0.88, 0.90, 1.00, 0.98, 0.95, 0.90, 0.80, 0.70, 0.60
    ])
    
    pv_profile = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0,
        0.9, 0.7, 0.5, 0.2, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ])
    
    ev_profile = np.zeros(24)
    ev_profile[11:15] = [0.8, 1.0, 1.0, 0.8]

    return load_profile, pv_profile, ev_profile


def run_simulation_for_sample(base_net, profiles, pv_penetration, ev_count):
    """
    Runs a 24h time-series simulation for a single scenario using a manual loop.
    """
    net = deepcopy(base_net)
    load_profile, pv_profile, ev_profile = profiles

    base_loads_p = net.load.p_mw.copy()
    base_loads_q = net.load.q_mvar.copy()

    total_peak_load_mw = base_loads_p.sum()
    total_pv_mw = pv_penetration * total_peak_load_mw
    total_ev_load_mw = ev_count * (EV_CHARGER_KW / 1000.0)

    # Add PV generators with reactive power control
    load_buses = net.load.bus.unique()
    new_pv_indices = []
    base_pv_p = {}
    base_pv_q = {} # Store base Q for PVs
    if total_pv_mw > 0 and len(load_buses) > 0:
        pv_per_bus_p = total_pv_mw / len(load_buses)
        # **FIXED**: Calculate reactive power based on power factor
        q_per_bus = pv_per_bus_p * np.tan(np.arccos(PV_POWER_FACTOR))
        for bus_idx in load_buses:
            idx = pp.create_sgen(net, bus=bus_idx, p_mw=pv_per_bus_p, q_mvar=q_per_bus, type='pv')
            new_pv_indices.append(idx)
            base_pv_p[idx] = pv_per_bus_p
            base_pv_q[idx] = q_per_bus
    
    # Add EV loads
    new_ev_load_indices = []
    base_ev_p = {}
    if total_ev_load_mw > 0 and len(load_buses) > 0:
        ev_load_per_bus = total_ev_load_mw / len(load_buses)
        for bus_idx in load_buses:
            idx = pp.create_load(net, bus=bus_idx, p_mw=ev_load_per_bus, q_mvar=0, name=f"EVs_bus_{bus_idx}")
            new_ev_load_indices.append(idx)
            base_ev_p[idx] = ev_load_per_bus

    # Manual simulation loop
    for t in range(TIMESTEPS):
        net.load.loc[base_loads_p.index, 'p_mw'] = base_loads_p * load_profile[t]
        net.load.loc[base_loads_q.index, 'q_mvar'] = base_loads_q * load_profile[t]
        
        for idx in new_pv_indices:
            # Scale both P and Q with the profile
            net.sgen.loc[idx, 'p_mw'] = base_pv_p[idx] * pv_profile[t]
            net.sgen.loc[idx, 'q_mvar'] = base_pv_q[idx] * pv_profile[t]
        
        for idx in new_ev_load_indices:
            net.load.loc[idx, 'p_mw'] = base_ev_p[idx] * ev_profile[t]

        try:
            pp.runpp(net)
        except Exception:
            return 'unsafe_convergence'

        if net.res_bus.vm_pu.min() < VM_MIN_PU or net.res_bus.vm_pu.max() > VM_MAX_PU:
            return 'unsafe_voltage'
        if not net.res_line.empty and net.res_line.loading_percent.max() > MAX_LINE_LOADING:
            return 'unsafe_overload'

    return 'safe'


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = "dataset_daytime_charging_pf_control.csv"
    output_path = os.path.join(script_dir, output_filename)
    
    print(f"--- Data Generation Script Started ---")
    print(f"Scenario: Daytime EV Charging (22kW) & PV with PF Control")
    print(f"Output will be saved to: {output_path}")

    print("\n1. Initializing grid and parameters...")
    net = get_grid()
    print("   - Grid loaded: MV Oberrhein network")

    print("\n2. Creating 24h load, PV, and EV profiles...")
    profiles = create_profiles()
    print("   - Profiles created successfully (Daytime EV charging).")

    print("\n3. Generating samples using Latin Hypercube Sampling...")
    sampler = qmc.LatinHypercube(d=2, seed=42)
    samples = sampler.random(n=NUM_SAMPLES)

    scaled_samples = qmc.scale(samples,
                               [PV_PENETRATION_RANGE[0], EV_COUNT_RANGE[0]],
                               [PV_PENETRATION_RANGE[1], EV_COUNT_RANGE[1]])
    scaled_samples[:, 1] = np.round(scaled_samples[:, 1]).astype(int)
    print(f"   - Generated {NUM_SAMPLES} samples.")

    print("\n4. Running simulations for all samples...")
    results = []
    for i in tqdm(range(NUM_SAMPLES), desc="Simulating scenarios"):
        pv_penetration = scaled_samples[i, 0]
        ev_count = int(scaled_samples[i, 1])
        
        label = run_simulation_for_sample(net, profiles, pv_penetration, ev_count)
        
        results.append({
            'pv_penetration': pv_penetration,
            'ev_count': ev_count,
            'label_reason': label,
            'label': 0 if label == 'safe' else 1
        })

    print("\n--- Simulation complete ---")
    
    print("\n5. Saving dataset...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    print(f"   - Dataset saved to {output_path}")
    safe_count = (results_df.label == 0).sum()
    unsafe_count = (results_df.label == 1).sum()
    print(f"   - Safe samples (label=0): {safe_count}")
    print(f"   - Unsafe samples (label=1): {unsafe_count}")
    
    if unsafe_count > 0:
        print("\nFailure reason breakdown for unsafe samples:")
        print(results_df[results_df.label == 1]['label_reason'].value_counts())


if __name__ == '__main__':
    main()
    print("\n--- Data Generation Script Finished ---")

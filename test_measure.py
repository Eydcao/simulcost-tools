#!/usr/bin/env python3
import json
import sys

print("Starting script...", flush=True)

# Load linear dataset
linear_dataset = 'dataset/hasegawa_mima_linear/successful/tasks.json'
print(f"Loading {linear_dataset}...", flush=True)

with open(linear_dataset, 'r') as f:
    data = json.load(f)

tasks = data['tasks']
print(f"Loaded {len(tasks)} tasks", flush=True)

# Find max N per profile
from collections import defaultdict
profile_max_N = defaultdict(lambda: {'N': 0, 'dt': None, 'cg_atol': None})

for task in tasks:
    profile = task['profile']
    target_param = task['target_parameter']
    optimal_value = task['results']['optimal_parameter_value']
    non_target = task['non_target_parameters']

    # Get N value
    if target_param == 'N':
        N = optimal_value
        dt = non_target.get('dt')
        cg_atol = non_target.get('cg_atol')
    else:
        N = non_target.get('N')
        dt = non_target.get('dt') if target_param != 'dt' else optimal_value
        cg_atol = non_target.get('cg_atol') if target_param != 'cg_atol' else optimal_value

    if N is not None and N > profile_max_N[profile]['N']:
        profile_max_N[profile]['N'] = N
        profile_max_N[profile]['dt'] = dt
        profile_max_N[profile]['cg_atol'] = cg_atol

print("\nMax N per profile (Linear):", flush=True)
for profile in sorted(profile_max_N.keys()):
    print(f"  {profile}: N={profile_max_N[profile]['N']}, dt={profile_max_N[profile]['dt']}, cg_atol={profile_max_N[profile]['cg_atol']}", flush=True)

print("\nDone!", flush=True)

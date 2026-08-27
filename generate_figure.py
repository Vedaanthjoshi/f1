import matplotlib.pyplot as plt
import numpy as np
import os

# Create data
laps = np.arange(40, 56)
# Net advantage starts negative (stay out is better), then crosses 0
net_advantage = [-1.5, -1.2, -0.8, -0.4, -0.1, 0.3, 0.8, 1.2, 1.6, 2.1, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(laps, net_advantage, color='#E10600', linewidth=3, marker='o', markersize=8, markerfacecolor='#E10600', markeredgewidth=1.5, markeredgecolor='white', label='Net Time Gained (s)')
ax.axhline(y=0, color='#FFFFFF', linestyle='-', alpha=0.3, linewidth=2)
ax.axvline(x=45, color='#FFD166', linestyle='--', linewidth=2, label='Crossover Lap (Tyres become slower)')

ax.set_title('Figure 4: Stay-Out vs Pit-Now Simulation (Net Undercut Advantage)', fontsize=16, pad=20)
ax.set_xlabel('Race Lap', fontsize=12)
ax.set_ylabel('Seconds Gained (+) or Lost (-) per lap', fontsize=12)

# Fill between for visual flair
ax.fill_between(laps, net_advantage, 0, where=(np.array(net_advantage) >= 0), color='#E10600', alpha=0.2)
ax.fill_between(laps, net_advantage, 0, where=(np.array(net_advantage) < 0), color='#888888', alpha=0.2)

ax.grid(True, alpha=0.1)
ax.legend(loc='upper left', fontsize=12)

plt.tight_layout()
save_path = 'c:/f1/figure4_undercut.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved figure to {save_path}")

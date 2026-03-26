"""
Plot Training curves with and without RMSNorm (lr = 1e-3)
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 12

# ── With RMSNorm (lr=1e-3) ──
norm_train_steps = [
    0, 50, 100, 150, 200, 250, 300, 350, 400, 450,
    500, 550, 600, 650, 700, 750, 800, 850, 900, 950,
    1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450,
    1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950,
    2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450,
    2500, 2550, 2600, 2650, 2700, 2750, 2800, 2850, 2900, 2950,
    3000, 3050, 3100, 3150, 3200, 3250, 3300, 3350, 3400, 3450,
    3500, 3550, 3600, 3650, 3700, 3750, 3800, 3850, 3900, 3950,
    4000, 4050, 4100, 4150, 4200, 4250, 4300, 4350, 4400, 4450,
    4500, 4550, 4600, 4650, 4700, 4750, 4800, 4850, 4900, 4950,
]
norm_train_loss = [
    9.2429, 5.6463, 4.0033, 3.3868, 3.1196, 2.8974, 2.6734, 2.6632, 2.5159, 2.6794,
    2.3862, 2.4233, 2.3963, 2.3105, 2.3689, 2.1617, 2.1786, 2.1850, 2.1003, 2.1265,
    2.1273, 2.1632, 2.1913, 2.1489, 2.2491, 2.1256, 2.1299, 2.1378, 2.1319, 2.1709,
    2.0305, 2.0207, 1.9319, 2.0071, 2.0098, 1.8667, 1.9054, 1.9941, 1.9105, 1.9684,
    2.0505, 1.9439, 1.8218, 2.1198, 1.8779, 1.8492, 1.9070, 1.8163, 1.8639, 1.7459,
    1.8637, 1.9650, 1.9894, 1.8833, 1.7899, 1.8197, 1.7817, 1.8473, 1.8833, 1.7840,
    1.8261, 1.7896, 1.8252, 1.7635, 1.8160, 1.7684, 1.8490, 1.7618, 1.6117, 1.7491,
    1.8422, 1.7838, 1.6829, 1.7229, 1.7097, 1.7285, 1.7165, 1.7647, 1.6668, 1.8013,
    1.8351, 1.6373, 1.7471, 1.8460, 1.7503, 1.6738, 1.6655, 1.7051, 1.6977, 1.6746,
    1.5992, 1.6855, 1.5697, 1.8062, 1.6054, 1.6244, 1.7294, 1.6930, 1.7070, 1.6449,
]

norm_val_steps = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
norm_val_loss = [9.2480, 2.4545, 2.1592, 2.0230, 1.9332, 1.8504, 1.7952, 1.7616, 1.7061, 1.6938]

# ── Without RMSNorm (lr=1e-3) ──
no_norm_train_steps = [
    0, 100, 200, 300, 400, 500, 600, 700, 800, 900,
    1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900,
    2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900,
    3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900,
    4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900,
]
no_norm_train_loss = [
    16.0362, 4.0887, 3.0920, 2.8566, 2.8558, 2.5419, 2.4240, 2.4412, 2.4059, 2.3047,
    2.3441, 2.1928, 2.2453, 2.2406, 2.1816, 2.2620, 2.1681, 1.9553, 2.1648, 1.9881,
    2.2317, 2.0123, 1.9997, 1.9833, 2.1001, 1.9780, 1.8748, 1.7636, 1.8638, 1.9381,
    1.8061, 1.8142, 1.8914, 1.7963, 1.7424, 1.8486, 1.7911, 1.8812, 1.8361, 1.7428,
    1.7083, 1.7193, 1.7464, 1.6514, 1.7585, 1.7857, 1.7501, 1.7735, 1.8481, 1.6764,
]

no_norm_val_steps = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
no_norm_val_loss = [17.1369, 2.6271, 2.3129, 2.1499, 2.0399, 1.9515, 1.8950, 1.8007, 1.7787, 1.7204]

# ── Plot ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- Left: Training Loss ---
ax1.plot(norm_train_steps, norm_train_loss,
         alpha=0.4, color='#2196F3', linewidth=0.8, label='With RMSNorm (raw)')
ax1.plot(no_norm_train_steps, no_norm_train_loss,
         alpha=0.4, color='#FF5722', linewidth=0.8, label='Without RMSNorm (raw)')

# Smoothed training loss (use val loss points as anchors for cleaner view)
ax1.plot(norm_val_steps, norm_val_loss,
         '-o', color='#1565C0', linewidth=2.2, markersize=5, label='With RMSNorm (val)')
ax1.plot(no_norm_val_steps, no_norm_val_loss,
         '-s', color='#D84315', linewidth=2.2, markersize=5, label='Without RMSNorm (val)')

ax1.set_xlabel('Training Step')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss & Validation Loss')
ax1.legend(loc='upper right', fontsize=10)
ax1.set_ylim(1.4, 10.0)
ax1.grid(True, alpha=0.3)

# --- Right: Validation Loss (zoomed in) ---
ax2.plot(norm_val_steps, norm_val_loss,
         '-o', color='#1565C0', linewidth=2.5, markersize=7, label='With RMSNorm')
ax2.plot(no_norm_val_steps, no_norm_val_loss,
         '-s', color='#D84315', linewidth=2.5, markersize=7, label='Without RMSNorm')

# Annotate final val loss
ax2.annotate(f'{norm_val_loss[-1]:.4f}',
             xy=(norm_val_steps[-1], norm_val_loss[-1]),
             xytext=(-60, 15), textcoords='offset points',
             fontsize=10, color='#1565C0', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.2))
ax2.annotate(f'{no_norm_val_loss[-1]:.4f}',
             xy=(no_norm_val_steps[-1], no_norm_val_loss[-1]),
             xytext=(-60, -20), textcoords='offset points',
             fontsize=10, color='#D84315', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#D84315', lw=1.2))

ax2.set_xlabel('Training Step')
ax2.set_ylabel('Validation Loss')
ax2.set_title('Validation Loss (Zoomed In)')
ax2.legend(loc='upper right', fontsize=11)
ax2.grid(True, alpha=0.3)

fig.suptitle('Training Curves With and Without RMSNorm (lr = 1e-3, 22.7M params)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('training_curves_rmsnorm.png', dpi=200, bbox_inches='tight')
plt.savefig('training_curves_rmsnorm.pdf', bbox_inches='tight')
print("Saved: training_curves_rmsnorm.png / .pdf")
plt.show()

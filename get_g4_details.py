import pandas as pd, numpy as np

g4_df = pd.read_csv('g4_pit_loss_results.csv')
green_df = g4_df[g4_df['classification'] == 'green'].copy()
green_losses = green_df['relative_loss_s'].values

p5 = float(np.percentile(green_losses, 5))
p95 = float(np.percentile(green_losses, 95))

print("=== RAW G4 GREEN PIT-LOSS SAMPLE ===")
print(f"Total green stops in G4: {len(green_df)}")
for idx, r in green_df.reset_index(drop=True).iterrows():
    print(f"#{idx+1:2d}: {r['driver']} L{int(r['in_lap']):2d}->L{int(r['out_lap']):2d} ({r['compound_before']}->{r['compound_after']}): {r['relative_loss_s']:.3f} s")

print(f"\n=== PERCENTILE THRESHOLDS ===")
print(f"5th Percentile:  {p5:.3f} s")
print(f"95th Percentile: {p95:.3f} s")

mask_kept = (green_losses >= p5) & (green_losses <= p95)
kept_df = green_df[mask_kept]
dropped_df = green_df[~mask_kept]

print(f"\n=== KEPT STOPS (n={len(kept_df)}) ===")
for idx, r in kept_df.iterrows():
    print(f"  {r['driver']} L{int(r['in_lap']):2d}->L{int(r['out_lap']):2d}: {r['relative_loss_s']:.3f} s")

print(f"\n=== DROPPED STOPS (n={len(dropped_df)}) ===")
for idx, r in dropped_df.iterrows():
    side = "BELOW 5th" if r['relative_loss_s'] < p5 else "ABOVE 95th"
    print(f"  {r['driver']} L{int(r['in_lap']):2d}->L{int(r['out_lap']):2d}: {r['relative_loss_s']:.3f} s ({side})")


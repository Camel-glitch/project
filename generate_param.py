import pandas as pd
import numpy as np

# Plages de paramètres
kappas = np.linspace(0.1, 10, 15)
thetas = np.linspace(0.01, 0.5, 10)
sigmas = np.linspace(0.1, 1.0, 10)

valid_points = []

for k in kappas:
    for t in thetas:
        for s in sigmas:
            # Condition de Feller : 2*kappa*theta > sigma^2
            if 2 * k * t > s**2:
                valid_points.append({'kappa': k, 'theta': t, 'sigma': s})

df = pd.DataFrame(valid_points)
df.to_csv('params_feller.csv', index=False, header=False)
print(f"{len(df)} points valides générés dans params_feller.csv")
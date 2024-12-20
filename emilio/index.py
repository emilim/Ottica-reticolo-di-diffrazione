import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def fit_lineare_doro(x, y, sigma_y):
    W = 1 / sigma_y**2  # Pesi
    Delta = np.sum(W) * np.sum(W * x**2) - (np.sum(W * x))**2
    m = (np.sum(W) * np.sum(W * x * y) - np.sum(W * x) * np.sum(W * y)) / Delta
    q = (np.sum(W * x**2) * np.sum(W * y) - np.sum(W * x) * np.sum(W * x * y)) / Delta
    sigma_m = np.sqrt(np.sum(W) / Delta)
    sigma_q = np.sqrt(np.sum(W * x**2) / Delta)
    return m, q, sigma_m, sigma_q

parser = argparse.ArgumentParser(description="File specificato")
parser.add_argument('filename', type=str, help='nome del file da leggere')

args = parser.parse_args()

# Lettura del file
df = pd.read_csv(args.filename, header=None)

d = 3.33e-6
theta0 = 10.69965 # gradi

ordini = np.array([-3, -2, -1, 1, 2, 3])
angoli = df.values.T[0] # - theta0 è già stato fatto
angoli_rad = (angoli/180)*np.pi
y = np.sin(angoli_rad)
s_y = y/100

m, q, s_m_, s_q = fit_lineare_doro(ordini, y, s_y)

lunghezza_donda = d * m
lunghezze_donda = d*y/ordini
print("Lunghezza d'onda:", lunghezza_donda*1e9, "nanometer")
print("Lunghezze d'onda:", lunghezze_donda*1e9, "nanometer")

plt.scatter(ordini, y)
plt.plot(ordini, ordini*m+q, label=f"Fit lineare\ny = {m:.5f}x + {q:.5f}")
plt.xlabel("Ordine")
plt.ylabel("sin(θ)")
plt.legend()
plt.grid()
plt.show()
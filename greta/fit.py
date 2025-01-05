import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fit_lineare_doro(x, y, sigma_y):
    W = 1 / sigma_y**2  # Pesi
    Delta = np.sum(W) * np.sum(W * x**2) - (np.sum(W * x))**2
    m = (np.sum(W) * np.sum(W * x * y) - np.sum(W * x) * np.sum(W * y)) / Delta
    q = (np.sum(W * x**2) * np.sum(W * y) - np.sum(W * x) * np.sum(W * x * y)) / Delta
    sigma_m = np.sqrt(np.sum(W) / Delta)
    sigma_q = np.sqrt(np.sum(W * x**2) / Delta)
    return m, q, sigma_m, sigma_q

file_path = 'dati_rosso.txt'

df = pd.read_csv(file_path, sep='\s+', header=None, names=['Ordine', 'Angolo'], skiprows=1, decimal=',')

d = 3.33e-6  #metri
s_d = 0.03e-6
theta0 = 10.69965 

ordini = df['Ordine'].astype(float).values[0:]  
angoli = df['Angolo'].astype(float).values[0:]  
angoli_rad = np.radians(angoli)  # Conversione in radianti
y = d*np.sin(angoli_rad)  

s_y = np.sqrt((np.power((y/d), 2)*np.power(s_d,2)) + (np.power(d*np.cos(angoli_rad), 2)*(np.power(0.000237,2))))

m, q, s_m, s_q = fit_lineare_doro(ordini, y, s_y)

lunghezza_donda = m #dal fit 
lunghezze_donda = y / ordini #dai dati
s_lunghezze_donda = np.sqrt((np.power((np.sin(angoli_rad)/ordini), 2)*np.power(s_d,2)) + (np.power((d*np.cos(angoli_rad)/ordini), 2)*(np.power(0.000336,2))))

chi_quadro_sper = np.sum(((y - m * ordini - q) / s_y)**2)



print(f"Lunghezza d'onda (da fit): {lunghezza_donda*1e6:.5f} μm")
print(f"Lunghezze d'onda (calcolate): {lunghezze_donda*1e6} μm")
print(f"s su lunghezze d'onda (calcolate):{s_lunghezze_donda*1e6} μm")
print(f"Chi quadro sperimentale: {chi_quadro_sper:.2f}")
print(f"Parametri del fit: m = {m:.15f} ± {s_m:.15f}, q = {q:.15f} ± {s_q:.15f}")
print("Ordini:", ordini)
print("Angoli:", angoli)
print("angoli rad", angoli_rad)
print("y:", y)
print("s_y:", s_y)
print(m)

plt.errorbar(ordini, y, c='b', yerr=s_y, fmt='None')
plt.scatter(ordini, y, label='Dati', color='red')  
plt.plot(ordini, m * ordini + q, label='Fit lineare')  
plt.xlabel("Ordine (m)")
plt.ylabel("dsin(θ-θ°)")
plt.legend()
plt.grid(True)
plt.title("Fit lineare - Analisi del colore rosso")
plt.savefig("fit_lineare_rosso.png", dpi=300)
plt.show()

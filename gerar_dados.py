# gera os dados falsos dos clientes (nao sao reais)

import numpy as np
import pandas as pd

rng = np.random.default_rng(13)
n = 400

horas = np.clip(rng.gamma(2.0, 3.0, n), 0, 30).round(1)
dias = np.clip(rng.exponential(8, n), 0, 60).round(0)

# quem assiste pouco e some do app tem mais chance de cancelar
z = 0.2 - 0.30 * horas + 0.10 * dias
p = 1 / (1 + np.exp(-z))
cancelou = rng.binomial(1, p)

df = pd.DataFrame({"horas_semana": horas, "dias_sem_abrir": dias, "cancelou": cancelou})
df.to_csv("clientes.csv", index=False)

print("clientes.csv gerado:", len(df), "clientes,", df["cancelou"].sum(), "cancelaram")

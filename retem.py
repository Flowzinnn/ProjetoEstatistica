# retem - grupo 13
# preve a chance do cliente cancelar a assinatura

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

df = pd.read_csv("clientes.csv")
print("clientes:", len(df), "| cancelaram:", df["cancelou"].sum())

X = df[["horas_semana", "dias_sem_abrir"]]
y = df["cancelou"]  # 1 cancelou, 0 ficou

# teste com regressao linear so pra mostrar que nao funciona
lin = LinearRegression().fit(df[["horas_semana"]], y)
p_lin = lin.predict(pd.DataFrame({"horas_semana": [25]}))[0]
print("\n[linear] 25h por semana ->", round(p_lin * 100), "% de chance de cancelar (???)")

lin2 = LinearRegression().fit(X, y)
p_lin2 = lin2.predict(pd.DataFrame({"horas_semana": [0], "dias_sem_abrir": [60]}))[0]
print("[linear] 0h e 60 dias sem abrir ->", round(p_lin2 * 100), "%")

def sigmoide(z):
    return 1 / (1 + np.exp(-z))

print("\nsigmoide(-5) =", round(sigmoide(-5), 3))
print("sigmoide(0)  =", sigmoide(0))
print("sigmoide(5)  =", round(sigmoide(5), 3))

# agora sim a logistica
modelo = LogisticRegression()
modelo.fit(X, y)

b0 = modelo.intercept_[0]
b1, b2 = modelo.coef_[0]
print(f"\nz = {b0:.2f} + ({b1:.2f})*horas + ({b2:.2f})*dias")

# e elevado ao coeficiente = quanto multiplica a chance
print("\nchance (odds) multiplica por:")
print(f"  +1 hora por semana -> x{np.exp(b1):.2f}")
print(f"  +1 dia sem abrir   -> x{np.exp(b2):.2f}")
print(f"  +10 dias sem abrir -> x{np.exp(b2 * 10):.1f}")

# clientes novos dos slides
novos = pd.DataFrame({
    "horas_semana": [1, 5, 15],
    "dias_sem_abrir": [25, 10, 1],
}, index=["Ana", "Carla", "Bruno"])

proba = modelo.predict_proba(novos)[:, 1]  # coluna 1 = chance de cancelar

print("\nclientes novos (limiar 0.5):")
for nome, p in zip(novos.index, proba):
    decisao = "manda desconto" if p >= 0.5 else "nao manda"
    print(f"  {nome:6} {p * 100:5.1f}%  -> {decisao}")

# testando outros limiares na base toda
p_todos = modelo.predict_proba(X)[:, 1]

print("\nlimiar | recebem desconto | cancelamentos pegos | desconto a toa | acerto")
for limiar in [0.3, 0.5, 0.7]:
    vai_cancelar = p_todos >= limiar
    pegos = (vai_cancelar & (y == 1)).sum()
    a_toa = (vai_cancelar & (y == 0)).sum()
    acerto = (vai_cancelar == y).mean()
    print(f"  {limiar}  | {vai_cancelar.sum():16} | {pegos:12} de {y.sum()} | {a_toa:14} | {acerto * 100:.1f}%")

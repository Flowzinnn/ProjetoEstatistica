# site do retem - roda em http://localhost:5000

import os
import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

pasta = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(pasta, "clientes.csv"))

# treina uma vez so quando o site abre
modelo = LogisticRegression()
modelo.fit(df[["horas_semana", "dias_sem_abrir"]], df["cancelou"])
b0 = modelo.intercept_[0]
b1, b2 = modelo.coef_[0]


def br(n, casas=2):
    # numero no formato brasileiro (virgula)
    return f"{n:.{casas}f}".replace(".", ",").replace("-", "−")


@app.route("/")
def index():
    horas = request.args.get("horas", type=float)
    dias = request.args.get("dias", type=float)
    limiar = request.args.get("limiar", default=0.5, type=float)

    resultado = None
    if horas is not None and dias is not None:
        cliente = pd.DataFrame({"horas_semana": [horas], "dias_sem_abrir": [dias]})
        p = modelo.predict_proba(cliente)[0, 1]
        z = b0 + b1 * horas + b2 * dias
        resultado = {
            "p": round(p * 100, 1),
            "p_txt": br(p * 100, 1),
            "z": br(z),
            "conta": f"{br(b0)} − {br(-b1)}·{br(horas, 1)} + {br(b2)}·{br(dias, 0)}",
            "cancela": p >= limiar,
        }

    return render_template(
        "index.html",
        horas=horas, dias=dias, limiar=limiar, r=resultado,
        formula=f"z = {br(b0)} − {br(-b1)}·horas + {br(b2)}·dias",
        total=len(df), cancelaram=int(df["cancelou"].sum()),
    )


if __name__ == "__main__":
    app.run(debug=True)

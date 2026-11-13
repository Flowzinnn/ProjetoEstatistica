# Retém

Seminário de regressão logística e classificação probabilística, Grupo 13.
Nicolas Wolf e Vinicius Antonio Lourençon.

Sistema que estima a probabilidade de um cliente cancelar a assinatura de um streaming,
a partir de duas variáveis: horas assistidas por semana e dias sem abrir o app.

Os dados são simulados (400 clientes gerados em Python), não são de uma empresa real.

## Arquivos

- `clientes.csv`: a base de dados
- `gerar_dados.py`: gera o `clientes.csv` (não precisa rodar, o arquivo já está aqui)
- `retem.py`: treina o modelo e mostra os resultados usados nos slides
- `app.py` + `templates/index.html`: site pra usar o Retém no navegador

## Como rodar

```
pip install -r requirements.txt
python retem.py
```

Pra abrir o site:

```
python app.py
```

Depois é só abrir http://localhost:5000 no navegador.

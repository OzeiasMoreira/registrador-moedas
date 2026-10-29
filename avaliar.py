"""
Avaliacao: compara a resposta do programa com o gabarito.

Uso:
    python avaliar.py fotos/ gabarito.csv
    python avaliar.py fotos_sinteticas/ gabarito_sinteticas.csv

O gabarito e um CSV com ';' e duas colunas: arquivo;moedas_centavos
Exemplo de linha:  foto01.jpg;100 50 25 5   (uma moeda de R$1, uma de 50c, ...)

Gera na pasta resultados/: tabela_acertos.csv, matriz_confusao.csv e matriz_confusao.png
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from moedas import processar, desenhar, reais

CLASSES = [5, 10, 25, 50, 100]
NENHUMA = 0   # "moeda que nao existe": falso positivo ou moeda nao detectada


def ler_gabarito(caminho):
    with open(caminho, newline="", encoding="utf-8") as fh:
        return {l["arquivo"]: [int(v) for v in l["moedas_centavos"].split()]
                for l in csv.DictReader(fh, delimiter=";")}


def pares_confusao(real, previsto):
    """Pareia moedas reais e previstas: iguais primeiro, sobras em seguida."""
    r, p = Counter(real), Counter(previsto)
    pares = []
    for v in set(r) & set(p):
        pares += [(v, v)] * min(r[v], p[v])
    sobra_r = list((r - p).elements())
    sobra_p = list((p - r).elements())
    for i in range(max(len(sobra_r), len(sobra_p))):
        pares.append((sobra_r[i] if i < len(sobra_r) else NENHUMA,
                      sobra_p[i] if i < len(sobra_p) else NENHUMA))
    return pares


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return
    pasta, gab = Path(sys.argv[1]), ler_gabarito(sys.argv[2])
    saida = Path("resultados")
    saida.mkdir(exist_ok=True)
    rotulos = CLASSES + [NENHUMA]
    conf = np.zeros((len(rotulos), len(rotulos)), int)
    linhas, ok_total, ok_qtd = [], 0, 0

    for nome, real in gab.items():
        img = cv2.imread(str(pasta / nome))
        if img is None:
            print("sem imagem:", nome)
            continue
        res = processar(img)
        cv2.imwrite(str(saida / (Path(nome).stem + "_resultado.png")), desenhar(res))
        prev = [m["valor"] for m in res["moedas"]]
        for a, b in pares_confusao(real, prev):
            conf[rotulos.index(a), rotulos.index(b)] += 1
        t_real, t_prev = sum(real), sum(prev)
        acertou = t_real == t_prev and sorted(real) == sorted(prev)
        ok_total += acertou
        ok_qtd += len(real) == len(prev)
        linhas.append([nome, reais(t_real), reais(t_prev), len(real), len(prev),
                       "SIM" if acertou else "NAO"])

    n = len(linhas)
    print("%-16s %-10s %-10s %5s %5s  %s" % ("arquivo", "real", "programa", "n_r", "n_p", "acertou"))
    for l in linhas:
        print("%-16s %-10s %-10s %5d %5d  %s" % tuple(l))
    print("\nFotos avaliadas: %d" % n)
    print("Acerto do valor total (e das moedas): %d/%d = %.1f%%" % (ok_total, n, 100 * ok_total / n))
    print("Acerto da contagem de moedas:         %d/%d = %.1f%%" % (ok_qtd, n, 100 * ok_qtd / n))
    ok_moedas = np.trace(conf[:-1, :-1])
    print("Moedas classificadas corretamente:    %d/%d = %.1f%%" % (
        ok_moedas, conf.sum(), 100 * ok_moedas / conf.sum()))

    nomes = [reais(c) if c else "nenhuma" for c in rotulos]
    print("\nMatriz de confusao (linhas = real, colunas = programa)")
    print("%9s" % "" + "".join("%9s" % t for t in nomes))
    for t, linha in zip(nomes, conf):
        print("%9s" % t + "".join("%9d" % v for v in linha))

    with open(saida / "tabela_acertos.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["arquivo", "total_real", "total_programa", "n_real", "n_programa", "acertou"])
        w.writerows(linhas)
    np.savetxt(saida / "matriz_confusao.csv", conf, fmt="%d", delimiter=";",
               header=";".join(nomes), comments="")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(conf, cmap="Blues")
    ax.set_xticks(range(len(nomes)), nomes, rotation=45)
    ax.set_yticks(range(len(nomes)), nomes)
    ax.set_xlabel("programa")
    ax.set_ylabel("real")
    for i in range(len(nomes)):
        for j in range(len(nomes)):
            ax.text(j, i, conf[i, j], ha="center", va="center",
                    color="white" if conf[i, j] > conf.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(saida / "matriz_confusao.png", dpi=150)


if __name__ == "__main__":
    main()

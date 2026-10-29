"""
Gera fotos SINTETICAS de moedas para testar o pipeline antes de ter as fotos reais.
(As fotos reais do grupo substituem estas: veja o LEIAME.txt.)

Cria fotos_sinteticas/ e gabarito_sinteticas.csv.
"""
import csv
import random
from pathlib import Path

import cv2
import numpy as np

# cores em HSV (OpenCV: H 0-179, S/V 0-255)
COR = {"cobre": (9, 175, 175), "dourada": (22, 150, 195), "prata": (20, 12, 190)}
MOEDAS = {5: (22.0, "cobre"), 10: (20.0, "dourada"), 25: (25.0, "dourada"),
          50: (23.0, "prata"), 100: (27.0, "bimetalica")}


def bgr(hsv):
    return tuple(int(v) for v in cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0, 0])


def desenha_moeda(img, cx, cy, r, valor):
    mat = MOEDAS[valor][1]
    if mat == "bimetalica":
        cor_anel, cor_miolo = bgr(COR["dourada"]), bgr(COR["prata"])
    else:
        cor_anel = cor_miolo = bgr(COR[mat])
    cv2.circle(img, (cx, cy), r, cor_anel, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), int(r * 0.60), cor_miolo, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, tuple(int(c * 0.6) for c in cor_anel), 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), int(r * 0.60), tuple(int(c * 0.7) for c in cor_miolo), 1, cv2.LINE_AA)
    # relevo: pequenos tracos simulando o desenho da moeda
    for _ in range(6):
        a = random.uniform(0, 6.28)
        d = random.uniform(0, r * 0.4)
        p = (int(cx + d * np.cos(a)), int(cy + d * np.sin(a)))
        cv2.circle(img, p, max(1, r // 14), tuple(int(c * 0.85) for c in cor_miolo), -1, cv2.LINE_AA)


def cena(i, rng):
    w, h = 1000, 750
    claro = (i % 5 == 4)                       # 1 em cada 5 fotos com fundo claro
    base = 215 if claro else 45
    img = np.full((h, w, 3), base, np.uint8)
    img = (img.astype(np.int16) + rng.normal(0, 4, img.shape)).clip(0, 255).astype(np.uint8)

    escala = rng.uniform(3.6, 4.4)             # px por mm
    n = int(rng.integers(3, 9))
    valores = [int(rng.choice(list(MOEDAS))) for _ in range(n)]
    colocadas = []
    encostadas = (i % 6 == 5)                  # algumas fotos com moedas encostadas
    for v in valores:
        r = int(MOEDAS[v][0] / 2 * escala)
        for _ in range(300):
            cx, cy = int(rng.integers(r + 10, w - r - 10)), int(rng.integers(r + 70, h - r - 10))
            folga = -2 if (encostadas and colocadas) else 12
            if all(np.hypot(cx - px, cy - py) > r + pr + folga for px, py, pr, _ in colocadas):
                break
        else:
            continue
        if encostadas and colocadas and rng.random() < 0.7:  # encosta na anterior
            px, py, pr, _ = colocadas[-1]
            ang = rng.uniform(0, 6.28)
            nx, ny = int(px + (r + pr + 1) * np.cos(ang)), int(py + (r + pr + 1) * np.sin(ang))
            ok = r + 10 < nx < w - r - 10 and r + 70 < ny < h - r - 10 and \
                all(np.hypot(nx - qx, ny - qy) >= r + qr for qx, qy, qr, _ in colocadas)
            if ok:
                cx, cy = nx, ny
        colocadas.append((cx, cy, r, v))

    for cx, cy, r, v in colocadas:
        desenha_moeda(img, cx, cy, r, v)

    # iluminacao desigual (gradiente) em parte das fotos
    if i % 3 == 2:
        grad = np.tile(np.linspace(0.65, 1.15, w), (h, 1))[..., None]
        img = (img * grad).clip(0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, sorted(v for *_, v in colocadas)


def main():
    rng = np.random.default_rng(42)
    random.seed(42)
    pasta = Path("fotos_sinteticas")
    pasta.mkdir(exist_ok=True)
    linhas = []
    for i in range(20):
        img, vals = cena(i, rng)
        nome = "sint%02d.png" % (i + 1)
        cv2.imwrite(str(pasta / nome), img)
        linhas.append([nome, " ".join(map(str, vals))])
    with open("gabarito_sinteticas.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["arquivo", "moedas_centavos"])
        w.writerows(linhas)
    print("geradas", len(linhas), "imagens em", pasta)


if __name__ == "__main__":
    main()

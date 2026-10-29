"""
Demonstracao passo a passo para a apresentacao.

Uso:
    python demo.py fotos/foto01.jpg
    python demo.py fotos_sinteticas/sint01.png

Abre uma janela com as etapas do pipeline. Teclas:
    ESPACO / seta direita = proxima etapa     seta esquerda = etapa anterior
    S = salva a montagem em resultados/demo_<foto>.png      Q ou ESC = sair
Se nao houver tela (ou cv2.imshow falhar), so salva a montagem.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

import moedas as M


def rotulo(img, texto):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(out, texto, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def etapas(img_bgr):
    img = M.redimensionar(img_bgr)
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fundo = M.estimar_fundo(cv2.GaussianBlur(cinza, (7, 7), 0))
    dif = cv2.absdiff(cv2.GaussianBlur(cinza, (7, 7), 0), fundo)
    mascara = M.mascara_primeiro_plano(cinza)
    res = M.processar(img_bgr)

    hough = img.copy()                       # circulos brutos do Hough
    for x, y, r in M.detectar_circulos(cinza, mascara):
        cv2.circle(hough, (int(x), int(y)), int(r), (0, 0, 255), 2)
        cv2.circle(hough, (int(x), int(y)), 3, (0, 0, 255), -1)

    cores = img.copy()                       # material por cor (HSV)
    paleta = {"cobre": (30, 80, 200), "dourada": (0, 200, 255),
              "prata": (200, 200, 200), "bimetalica": (255, 120, 0)}
    for m in res["moedas"]:
        c = (int(m["x"]), int(m["y"]))
        cv2.circle(cores, c, int(m["r"]), paleta[m["material"]], 4)
        cv2.putText(cores, m["material"], (c[0] - 40, c[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3)
        cv2.putText(cores, m["material"], (c[0] - 40, c[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)

    g3 = lambda g: cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return [
        rotulo(img, "1. Foto original (redimensionada)"),
        rotulo(g3(cinza), "2. Escala de cinza"),
        rotulo(g3(cv2.normalize(dif, None, 0, 255, cv2.NORM_MINMAX)), "3. Diferenca para o fundo estimado"),
        rotulo(g3(mascara), "4. Mascara (limiar + morfologia)"),
        rotulo(hough, "5. Hough: circulos encontrados"),
        rotulo(cores, "6. Cor HSV -> material"),
        rotulo(M.desenhar(res), "7. Resultado: valor de cada moeda e total"),
    ]


def montagem(lista):
    h, w = lista[0].shape[:2]
    peq = [cv2.resize(e, (w // 2, h // 2)) for e in lista]
    while len(peq) % 4:
        peq.append(np.zeros_like(peq[0]))
    linhas = [np.hstack(peq[i:i + 4]) for i in range(0, len(peq), 4)]
    return np.vstack(linhas)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return
    caminho = Path(sys.argv[1])
    img = cv2.imread(str(caminho))
    if img is None:
        print("nao consegui abrir", caminho)
        return
    lista = etapas(img)
    Path("resultados").mkdir(exist_ok=True)
    destino = Path("resultados") / ("demo_%s.png" % caminho.stem)
    cv2.imwrite(str(destino), montagem(lista))
    print("montagem salva em", destino)

    i = 0
    try:
        janela = "Caixa registradora de moedas"
        while True:
            cv2.imshow(janela, lista[i])
            k = cv2.waitKeyEx(100)
            if cv2.getWindowProperty(janela, cv2.WND_PROP_VISIBLE) < 1:
                break                         # janela fechada no X
            if k == -1:
                continue                      # nenhuma tecla ainda
            if k in (ord("q"), 27):
                break
            if k in (32, 2555904):            # espaco / seta direita
                if i == len(lista) - 1:
                    break                     # passou da ultima etapa: sai
                i += 1
            elif k == 2424832:                # seta esquerda
                i = max(i - 1, 0)
            elif k in (ord("s"), ord("S")):
                cv2.imwrite(str(destino), montagem(lista))
        cv2.destroyAllWindows()
    except cv2.error:
        print("(sem janela disponivel: so a montagem foi salva)")


if __name__ == "__main__":
    main()

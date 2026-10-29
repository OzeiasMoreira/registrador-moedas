"""
Caixa registradora de moedas - Visao Computacional (Avaliacao 2)

Recebe fotos de moedas brasileiras sobre uma mesa e calcula o valor total.

Pipeline (cada etapa e explicada no relatorio):
  1. Redimensiona a foto e converte para cinza
  2. Filtro da mediana para tirar ruido, mantendo as bordas
  3. Diferenca para o fundo estimado + limiarizacao de Otsu + morfologia -> mascara de "primeiro plano"
  4. CONCEITO NOVO: Transformada de Hough para circulos (cv2.HoughCircles)
     encontra cada moeda e mede o raio dela
  5. CONCEITO NOVO: segmentacao por cor no espaco HSV mede a cor do centro e da
     borda de cada moeda -> descobre o "material" (cobre, dourada, prata, bimetalica)
  6. Raio (em mm, usando a escala da foto) + material -> valor da moeda
  7. Soma os valores e desenha o resultado na imagem

Uso:
    python moedas.py fotos/ --saida resultados/
    python moedas.py fotos/foto01.jpg --debug
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Dados das moedas (familia de 2002/2003): valor em centavos -> (diametro mm, material)
# ---------------------------------------------------------------------------
MOEDAS = {
    5: (22.0, "cobre"),
    10: (20.0, "dourada"),
    25: (25.0, "dourada"),
    50: (23.0, "prata"),
    100: (27.0, "bimetalica"),
}

# ---------------------------------------------------------------------------
# Parametros (ajustados por tentativa e erro; ver secao Metodologia do relatorio)
# ---------------------------------------------------------------------------
LARGURA = 1000            # largura padrao da imagem de trabalho (px)
# (a faixa de raios das moedas e calculada sozinha a partir da mascara)
HOUGH_DP = 1.2            # resolucao do acumulador (1 = igual a imagem)
HOUGH_PARAM1 = 120        # limiar alto do Canny interno
HOUGH_PARAM2 = 40         # votos minimos para aceitar um circulo (menor = mais circulos)
FATOR_OTSU = 0.4          # fracao do limiar de Otsu usada na mascara
LIMIAR_MIN = 12           # limiar minimo (acima do ruido da camera)
FRACAO_FUNDO_MIN = 0.70   # % do circulo que precisa estar no primeiro plano
SAT_PRATA = 55            # saturacao (0-255) abaixo disso = metal "cinza/prata"
DELTA_SAT_BIMETAL = 45    # diferenca de saturacao centro x borda = moeda bimetalica
HUE_COBRE_MAX = 14        # matiz (0-179) abaixo disso = cobre; acima = dourada
ESCALA_PADRAO = 4.0       # px por mm, usada se nao der para calibrar sozinho


# ---------------------------------------------------------------------------
# Etapas do pipeline
# ---------------------------------------------------------------------------
def redimensionar(img, largura=LARGURA):
    h, w = img.shape[:2]
    return cv2.resize(img, (largura, int(h * largura / w)), interpolation=cv2.INTER_AREA)


def estimar_fundo(cinza):
    """Estima o fundo (inclusive gradiente de luz) com uma mediana bem grande.

    As moedas ocupam so uma parte da foto, entao a mediana de uma janela grande
    "enxerga" o fundo. Reduz a imagem antes para ficar rapido."""
    h, w = cinza.shape
    pequena = cv2.resize(cinza, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    k = (w // 4 // 3) | 1                                   # janela impar ~ 1/3 da largura
    fundo = cv2.medianBlur(pequena, k)
    return cv2.resize(fundo, (w, h), interpolation=cv2.INTER_LINEAR)


def mascara_primeiro_plano(cinza):
    """Diferenca para o fundo + Otsu + morfologia. 255 = moeda, 0 = fundo.

    Funciona com fundo claro ou escuro e com luz desigual, porque olha o quanto
    cada pixel DIFERE do fundo local, e nao o brilho absoluto."""
    suave = cv2.GaussianBlur(cinza, (7, 7), 0)
    dif = cv2.absdiff(suave, estimar_fundo(suave))
    t_otsu, _ = cv2.threshold(dif, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu sozinho acompanha as moedas mais contrastantes (cobre/dourada) e perde as
    # prateadas, que se parecem com o fundo. Por isso usamos uma fracao do limiar.
    limiar = max(LIMIAR_MIN, FATOR_OTSU * t_otsu)
    _, bin_ = cv2.threshold(dif, limiar, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_OPEN, k)    # tira pontinhos de ruido
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, k)   # fecha buracos (reflexos)
    # preenche o interior de cada objeto (desenhos/relevos escuros da moeda viram "buracos")
    cont, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cheio = np.zeros_like(bin_)
    cv2.drawContours(cheio, cont, -1, 255, -1)
    return cheio


def faixa_de_raios(mascara):
    """Descobre a faixa de raios das moedas olhando a propria mascara.

    A transformada de distancia diz, para cada pixel, a que distancia ele esta do
    fundo. O maior valor de cada objeto e o raio da maior moeda inscrita nele.
    Assim funciona tanto com moedas pequenas quanto numa foto de perto."""
    dist = cv2.distanceTransform(mascara, cv2.DIST_L2, 5)
    n, rotulos = cv2.connectedComponents(mascara)
    picos = [dist[rotulos == i].max() for i in range(1, n)]
    picos = [p for p in picos if p >= mascara.shape[1] * 0.008]   # ignora sujeira
    if not picos:
        return None
    return max(5, int(0.6 * min(picos))), int(1.25 * max(picos)) + 1


def detectar_circulos(cinza, mascara):
    """Hough para circulos; descarta circulos que caem no fundo."""
    h, w = cinza.shape
    faixa = faixa_de_raios(mascara)
    if faixa is None:
        return []
    r_min, r_max = faixa
    suave = cv2.medianBlur(cinza, 5)
    circ = cv2.HoughCircles(suave, cv2.HOUGH_GRADIENT, dp=HOUGH_DP,
                            minDist=r_min * 1.8, param1=HOUGH_PARAM1,
                            param2=HOUGH_PARAM2, minRadius=r_min, maxRadius=r_max)
    if circ is None:
        return []
    aceitos = []
    for x, y, r in circ[0]:
        disco = np.zeros_like(mascara)
        cv2.circle(disco, (int(x), int(y)), int(r), 255, -1)
        dentro = cv2.countNonZero(cv2.bitwise_and(mascara, disco)) / max(1, cv2.countNonZero(disco))
        if dentro >= FRACAO_FUNDO_MIN:
            aceitos.append((float(x), float(y), float(r)))
    return aceitos


def medir_raio(mascara, x, y, r_hough):
    """Refina o raio: dispara 72 raios a partir do centro ate sair da mascara.

    O Hough acerta bem o centro, mas o raio pode variar alguns pixels (e as vezes
    ele prende no circulo interno de moedas bimetalicas). A mediana dos 72 raios
    ignora os poucos que atravessam uma moeda vizinha encostada."""
    h, w = mascara.shape
    ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    ini = int(r_hough * 0.6)          # ignora buracos de reflexo perto do centro
    passos = np.arange(ini, int(r_hough * 1.6))
    xs = np.clip((x + np.outer(np.cos(ang), passos)).astype(int), 0, w - 1)
    ys = np.clip((y + np.outer(np.sin(ang), passos)).astype(int), 0, h - 1)
    dentro = mascara[ys, xs] > 0
    # primeiro passo em que o raio sai da moeda
    saiu = np.where(dentro.all(axis=1), len(passos), np.argmin(dentro, axis=1))
    return float(ini + np.median(saiu))


def remover_duplicadas(moedas):
    """Depois de refinar o raio, dois circulos com quase o mesmo centro sao a mesma moeda."""
    mantidas = []
    for m in sorted(moedas, key=lambda m: -m["r"]):
        if all(np.hypot(m["x"] - k["x"], m["y"] - k["y"]) > 0.5 * k["r"] for k in mantidas):
            mantidas.append(m)
    return mantidas


def cores_da_moeda(hsv, x, y, r):
    """Mediana de H,S,V no centro e no anel externo da moeda."""
    forma = hsv.shape[:2]
    centro = np.zeros(forma, np.uint8)
    cv2.circle(centro, (int(x), int(y)), int(r * 0.45), 255, -1)
    anel = np.zeros(forma, np.uint8)
    cv2.circle(anel, (int(x), int(y)), int(r * 0.90), 255, -1)
    cv2.circle(anel, (int(x), int(y)), int(r * 0.70), 0, -1)
    c = np.median(hsv[centro > 0], axis=0)
    a = np.median(hsv[anel > 0], axis=0)
    return {"h_c": c[0], "s_c": c[1], "v_c": c[2], "h_a": a[0], "s_a": a[1], "v_a": a[2]}


def material(f):
    """Regras sobre HSV -> cobre, dourada, prata ou bimetalica."""
    if abs(f["s_c"] - f["s_a"]) > DELTA_SAT_BIMETAL:
        return "bimetalica"
    if f["s_c"] < SAT_PRATA and f["s_a"] < SAT_PRATA:
        return "prata"
    return "cobre" if f["h_a"] <= HUE_COBRE_MAX else "dourada"


def estimar_escala(moedas, escala_manual=None):
    """px por mm. Usa moedas cujo material identifica o valor sozinho."""
    if escala_manual:
        return escala_manual
    unicos = {"cobre": 22.0, "prata": 23.0, "bimetalica": 27.0}
    est = [2 * m["r"] / unicos[m["material"]] for m in moedas if m["material"] in unicos]
    return float(np.median(est)) if est else ESCALA_PADRAO


def valor_da_moeda(mat, diam_mm):
    """Entre as moedas do mesmo material, escolhe a de diametro mais proximo."""
    cand = [(abs(d - diam_mm), v) for v, (d, m) in MOEDAS.items() if m == mat]
    return min(cand)[1]


def reais(centavos):
    return "R$ %d,%02d" % divmod(centavos, 100)


# ---------------------------------------------------------------------------
# Processamento de uma imagem
# ---------------------------------------------------------------------------
def processar(img_bgr, escala_manual=None):
    img = redimensionar(img_bgr)
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mascara = mascara_primeiro_plano(cinza)

    moedas = []
    for x, y, r_h in detectar_circulos(cinza, mascara):
        r = medir_raio(mascara, x, y, r_h)
        f = cores_da_moeda(hsv, x, y, r)
        moedas.append({"x": x, "y": y, "r": r, "r_hough": r_h, "feat": f,
                       "material": material(f)})
    moedas = remover_duplicadas(moedas)

    escala = estimar_escala(moedas, escala_manual)
    for m in moedas:
        m["diam_mm"] = 2 * m["r"] / escala
        m["valor"] = valor_da_moeda(m["material"], m["diam_mm"])
    moedas.sort(key=lambda m: (round(m["y"] / 80), m["x"]))
    total = sum(m["valor"] for m in moedas)
    return {"img": img, "mascara": mascara, "moedas": moedas, "escala": escala, "total": total}


def desenhar(res):
    out = res["img"].copy()
    for m in res["moedas"]:
        c = (int(m["x"]), int(m["y"]))
        cv2.circle(out, c, int(m["r"]), (0, 255, 0), 3)
        cv2.circle(out, c, 3, (0, 0, 255), -1)
        txt = "%.2f" % (m["valor"] / 100)
        cv2.putText(out, txt, (c[0] - 28, c[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(out, txt, (c[0] - 28, c[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    faixa = "%s  (%d moedas)" % (reais(res["total"]), len(res["moedas"]))
    cv2.rectangle(out, (0, 0), (out.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(out, faixa, (15, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    return out


def listar_imagens(entrada):
    p = Path(entrada)
    if p.is_file():
        return [p]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(f for f in p.iterdir() if f.suffix.lower() in exts)


def main():
    ap = argparse.ArgumentParser(description="Caixa registradora de moedas")
    ap.add_argument("entrada", help="foto ou pasta com fotos")
    ap.add_argument("--saida", default="resultados", help="pasta das imagens de resultado")
    ap.add_argument("--escala", type=float, default=None, help="px por mm (se souber)")
    ap.add_argument("--debug", action="store_true", help="mostra medidas HSV de cada moeda")
    a = ap.parse_args()

    saida = Path(a.saida)
    saida.mkdir(parents=True, exist_ok=True)
    linhas = []
    for caminho in listar_imagens(a.entrada):
        img = cv2.imread(str(caminho))
        if img is None:
            print("nao consegui abrir", caminho)
            continue
        res = processar(img, a.escala)
        cv2.imwrite(str(saida / (caminho.stem + "_resultado.png")), desenhar(res))
        previsto = " ".join(str(m["valor"]) for m in res["moedas"])
        print("%-22s %-12s %2d moedas | %s" % (caminho.name, reais(res["total"]),
                                                len(res["moedas"]), previsto))
        if a.debug:
            print("   escala = %.2f px/mm" % res["escala"])
            for m in res["moedas"]:
                f = m["feat"]
                print("   r=%5.1f d=%4.1fmm %-10s H/S/V centro=%3d/%3d/%3d anel=%3d/%3d/%3d -> %d"
                      % (m["r"], m["diam_mm"], m["material"], f["h_c"], f["s_c"], f["v_c"],
                         f["h_a"], f["s_a"], f["v_a"], m["valor"]))
        linhas.append([caminho.name, previsto, res["total"]])
    with open(saida / "previsoes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["arquivo", "moedas_previstas_centavos", "total_centavos"])
        w.writerows(linhas)


if __name__ == "__main__":
    main()

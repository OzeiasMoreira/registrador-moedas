"""
Caixa registradora ao vivo: usa a camera (webcam ou celular como webcam).

Uso:
    python camera.py              # camera 0
    python camera.py --camera 1   # outra camera
    python camera.py --video teste.mp4

Teclas:
    ESPACO = tira a foto (salva em fotos/ e o resultado em resultados/)
    Q ou ESC = sair
O total mostrado e o valor mais frequente nos ultimos quadros, para nao ficar
piscando quando uma moeda e detectada so em alguns quadros.
"""
import argparse
import time
from collections import Counter, deque
from pathlib import Path

import cv2

import moedas as M


def abrir(args):
    if args.video:
        return cv2.VideoCapture(args.video)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)   # CAP_DSHOW: mais estavel no Windows
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    return cap


def main():
    ap = argparse.ArgumentParser(description="Caixa registradora de moedas ao vivo")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", default=None, help="usa um arquivo de video no lugar da camera")
    ap.add_argument("--escala", type=float, default=None, help="px por mm (se souber)")
    args = ap.parse_args()

    cap = abrir(args)
    if not cap.isOpened():
        print("Nao consegui abrir a camera %d. Tente --camera 1 ou feche outros programas "
              "que usam a camera." % args.camera)
        return

    janela = "Caixa registradora de moedas"
    historico = deque(maxlen=7)       # totais dos ultimos quadros processados
    ultimo = None
    n = 0
    t0 = time.time()
    while True:
        ok, quadro = cap.read()
        if not ok:
            break
        n += 1
        if n % 3 == 0:                # processa 1 a cada 3 quadros (o Hough e pesado)
            ultimo = M.processar(quadro, args.escala)
            historico.append((ultimo["total"], len(ultimo["moedas"])))
        if ultimo is None:
            cv2.imshow(janela, M.redimensionar(quadro))
        else:
            vista = M.desenhar(ultimo)
            total, qtd = Counter(historico).most_common(1)[0][0]
            cv2.rectangle(vista, (0, 0), (vista.shape[1], 50), (0, 0, 0), -1)
            cv2.putText(vista, "%s  (%d moedas)" % (M.reais(total), qtd), (15, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
            cv2.putText(vista, "ESPACO = foto | Q = sair", (vista.shape[1] - 330, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow(janela, vista)

        k = cv2.waitKey(1) & 0xFF
        if cv2.getWindowProperty(janela, cv2.WND_PROP_VISIBLE) < 1 or k in (ord("q"), 27):
            break
        if k == 32:                   # espaco: salva a foto e o resultado
            nome = time.strftime("camera_%Y%m%d_%H%M%S")
            Path("fotos").mkdir(exist_ok=True)
            Path("resultados").mkdir(exist_ok=True)
            cv2.imwrite("fotos/%s.png" % nome, quadro)
            res = M.processar(quadro, args.escala)
            cv2.imwrite("resultados/%s_resultado.png" % nome, M.desenhar(res))
            print("foto salva: fotos/%s.png -> %s (%d moedas)" % (
                nome, M.reais(res["total"]), len(res["moedas"])))
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

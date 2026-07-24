import argparse
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--saida", default="frames")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--inicio", type=float, default=0.0)
    ap.add_argument("--fim", type=float, default=None)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / fps

    ini = args.inicio
    fim = args.fim if args.fim is not None else dur
    fim = min(fim, dur)

    tempos = [ini + (fim - ini) * i / max(args.n - 1, 1) for i in range(args.n)]

    destino = Path(args.saida)
    destino.mkdir(parents=True, exist_ok=True)

    salvos = 0
    for i, t in enumerate(tempos):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        nome = destino / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(nome), frame)
        salvos += 1
    cap.release()

    print(f"Video: {dur:.1f}s @ {fps:.0f}fps")
    print(f"{salvos} frames salvos em {destino}/ (amostrados uniformemente entre {ini:.1f}s e {fim:.1f}s)")
    print("Anote cada frame em formato YOLO (uma pasta de .txt com o mesmo nome do .jpg).")


if __name__ == "__main__":
    main()

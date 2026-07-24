"""
python contagem_carros.py --video rua_08h.mp4 --janela "08:00"
python contagem_carros.py --video rua_08h.mp4 --janela "08:00" --preview
"""

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

import cv2
from ultralytics import YOLO

VEICULOS = {2: "carro", 3: "moto", 5: "onibus", 7: "caminhao"}

# relativo (0-1) pra nao depender da resolucao do video
LINHA = ((0.05, 0.78), (0.95, 0.78))


def lado_da_linha(ponto, a, b):
    return (b[0] - a[0]) * (ponto[1] - a[1]) - (b[1] - a[1]) * (ponto[0] - a[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--janela", required=True, help='Hora do inicio da gravacao, ex: "08:00"')
    ap.add_argument("--saida", default="contagens.csv")
    ap.add_argument("--modelo", default="yolov8n.pt", help="n=rapido, m=preciso")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--preview", action="store_true", help="Mostra o video anotado (bom para gravar a tela)")
    args = ap.parse_args()

    modelo = YOLO(args.modelo)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    a = (LINHA[0][0] * largura, LINHA[0][1] * altura)
    b = (LINHA[1][0] * largura, LINHA[1][1] * altura)
    inicio = datetime.strptime(args.janela, "%H:%M")

    lado_anterior = {}
    ja_contados = set()
    registros = []

    stream = modelo.track(
        source=args.video,
        tracker="bytetrack.yaml",
        classes=list(VEICULOS),
        conf=args.conf,
        persist=True,
        stream=True,
        verbose=False,
    )

    for n_frame, r in enumerate(stream):
        if r.boxes.id is None:
            continue

        for box, tid, cls in zip(r.boxes.xyxy.cpu(), r.boxes.id.int().cpu(), r.boxes.cls.int().cpu()):
            tid, cls = int(tid), int(cls)
            x1, y1, x2, y2 = box.tolist()
            centro = ((x1 + x2) / 2, (y1 + y2) / 2)

            lado = lado_da_linha(centro, a, b)
            anterior = lado_anterior.get(tid)
            lado_anterior[tid] = lado

            if anterior is not None and anterior * lado < 0 and tid not in ja_contados:
                ja_contados.add(tid)
                registros.append({
                    "horario": (inicio + timedelta(seconds=n_frame / fps)).strftime("%H:%M:%S"),
                    "id": tid,
                    "classe": VEICULOS[cls],
                    "sentido": "descendo" if lado > 0 else "subindo",
                    "janela": args.janela,
                })

        if args.preview:
            frame = r.plot()
            cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 255, 255), 3)
            cv2.putText(frame, f"Total: {len(ja_contados)}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
            cv2.imshow("contagem", frame)
            if cv2.waitKey(1) == 27:
                break

    cv2.destroyAllWindows()

    saida = Path(args.saida)
    novo = not saida.exists()
    with saida.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["horario", "id", "classe", "sentido", "janela"])
        if novo:
            w.writeheader()
        w.writerows(registros)

    print(f"Janela {args.janela}: {len(registros)} veiculos contados -> {saida}")
    for c in VEICULOS.values():
        n = sum(1 for r in registros if r["classe"] == c)
        if n:
            print(f"  {c:10} {n:4}")


if __name__ == "__main__":
    main()

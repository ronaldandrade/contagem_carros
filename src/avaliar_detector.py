import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nucleo_avaliacao import (avaliar_classe, casar_por_imagem,
                              metricas_por_limiar)

CLASSES_COCO = {2: "carro", 3: "moto", 5: "onibus", 7: "caminhao"}
# Mapeamento das classes do data.yaml (Roboflow, ordem alfabetica: bus, car, motocycle)
# para os IDs de classe do COCO usados pelo modelo YOLO pre-treinado.
ROBOFLOW_PARA_COCO = {0: 5, 1: 2, 2: 3}

# Classes de um modelo treinado neste dataset (mesma ordem do data.yaml), sem
# necessidade de mapear para o COCO: o GT ja usa esses IDs diretamente.
CLASSES_CUSTOM = {0: "onibus", 1: "carro", 2: "moto"}
GT_IDENTIDADE = {0: 0, 1: 1, 2: 2}

AZUL, LARANJA, VERDE, CINZA, TEXTO = "#2f6fdd", "#e07b39", "#2e9e6b", "#c9ced6", "#1f2328"


def ler_gt_yolo(caminho_txt, largura, altura, classes, mapa_gt):
    caixas = {c: [] for c in classes}
    if not Path(caminho_txt).exists():
        return caixas
    for linha in Path(caminho_txt).read_text().strip().splitlines():
        if not linha.strip():
            continue
        partes = linha.split()
        cls = mapa_gt.get(int(partes[0]))
        if cls is None:
            continue
        cx, cy, w, h = map(float, partes[1:5])
        x1 = (cx - w / 2) * largura
        y1 = (cy - h / 2) * altura
        x2 = (cx + w / 2) * largura
        y2 = (cy + h / 2) * altura
        if cls in caixas:
            caixas[cls].append([x1, y1, x2, y2])
    return {c: np.array(v, dtype=float).reshape(-1, 4) for c, v in caixas.items()}


def rodar_deteccoes(modelo_path, imagens, conf_min, classes):
    from ultralytics import YOLO
    modelo = YOLO(modelo_path)
    preds_por_img = {c: [] for c in classes}
    tamanhos = []
    for img in imagens:
        r = modelo(str(img), conf=conf_min, classes=list(classes), verbose=False)[0]
        h, w = r.orig_shape
        tamanhos.append((img, w, h))
        caixas = {c: [] for c in classes}
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for b, cf, cl in zip(xyxy, conf, cls):
                if cl in caixas:
                    caixas[cl].append([b[0], b[1], b[2], b[3], cf])
        for c in classes:
            preds_por_img[c].append(np.array(caixas[c], dtype=float).reshape(-1, 5))
    return preds_por_img, tamanhos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagens", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--modelo", default="yolov8n.pt")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf-min", type=float, default=0.001)
    ap.add_argument("--classes", choices=["coco", "custom"], default="coco",
                     help="'coco' para modelos pre-treinados no COCO; 'custom' para um modelo "
                          "treinado neste dataset (classes bus/car/motocycle do data.yaml)")
    args = ap.parse_args()

    if args.classes == "custom":
        CLASSES, mapa_gt = CLASSES_CUSTOM, GT_IDENTIDADE
    else:
        CLASSES, mapa_gt = CLASSES_COCO, ROBOFLOW_PARA_COCO

    imagens = sorted(Path(args.imagens).glob("*.jpg")) + sorted(Path(args.imagens).glob("*.png"))
    if not imagens:
        raise SystemExit(f"Nenhuma imagem em {args.imagens}")

    preds_por_img, tamanhos = rodar_deteccoes(args.modelo, imagens, args.conf_min, CLASSES)

    gts_por_img = {c: [] for c in CLASSES}
    for img, w, h in tamanhos:
        txt = Path(args.labels) / (img.stem + ".txt")
        gt = ler_gt_yolo(txt, w, h, CLASSES, mapa_gt)
        for c in CLASSES:
            gts_por_img[c].append(gt[c])

    resultados = {}
    marcas_todas = []
    n_gt_todas = 0
    for c, nome in CLASSES.items():
        res = avaliar_classe(preds_por_img[c], gts_por_img[c], args.iou)
        resultados[nome] = res
        for preds, gts in zip(preds_por_img[c], gts_por_img[c]):
            marcas, n_gt = casar_por_imagem(preds, gts, args.iou)
            marcas_todas.extend(marcas)
            n_gt_todas += n_gt

    aps = {nome: r["ap"] for nome, r in resultados.items() if r["n_gt"] > 0}
    mAP = float(np.mean(list(aps.values()))) if aps else 0.0

    print(f"Avaliacao em {len(imagens)} frames | IoU = {args.iou}")
    print(f"{'classe':10} {'AP':>7} {'n_GT':>6}")
    for nome, r in resultados.items():
        marca = "" if r["n_gt"] > 0 else "  (sem GT, ignorada)"
        print(f"{nome:10} {r['ap']:7.3f} {r['n_gt']:6d}{marca}")
    print(f"\nmAP@{args.iou} = {mAP:.3f}")

    limiares = np.round(np.arange(0.05, 0.96, 0.05), 2)
    tabela = metricas_por_limiar(marcas_todas, n_gt_todas, limiares)
    df = pd.DataFrame(tabela)
    df.to_csv("metricas_por_limiar.csv", index=False)
    melhor = df.loc[df["f1"].idxmax()]
    print(f"\nMelhor F1 = {melhor['f1']:.3f} no limiar de confianca {melhor['limiar']:.2f} "
          f"(precisao {melhor['precisao']:.2f}, recall {melhor['recall']:.2f})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1, 1.05]})

    cores = [AZUL, LARANJA, VERDE, "#8e44ad"]
    for (nome, r), cor in zip(resultados.items(), cores):
        if r["n_gt"] > 0 and len(r["recall"]):
            ax1.plot(r["recall"], r["precision"], "-", color=cor, lw=2,
                     label=f"{nome} (AP={r['ap']:.2f})")
    ax1.set_xlabel("Recall", fontsize=11)
    ax1.set_ylabel("Precisão", fontsize=11)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.02)
    ax1.set_title(f"Curva precisão-recall  |  mAP@{args.iou} = {mAP:.2f}", fontsize=12, loc="left", pad=12)
    ax1.legend(frameon=False, fontsize=9, loc="lower left")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(df["limiar"], df["precisao"], "-o", color=AZUL, lw=2, markersize=4, label="Precisão")
    ax2.plot(df["limiar"], df["recall"], "-o", color=LARANJA, lw=2, markersize=4, label="Recall")
    ax2.plot(df["limiar"], df["f1"], "-o", color=VERDE, lw=2.4, markersize=4, label="F1")
    ax2.axvline(melhor["limiar"], color=CINZA, ls="--", lw=1.4)
    ax2.set_xlabel("Limiar de confiança", fontsize=11)
    ax2.set_ylabel("Métrica", fontsize=11)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02)
    ax2.set_title(f"Escolha do limiar: melhor F1 em {melhor['limiar']:.2f}", fontsize=12, loc="left", pad=12)
    ax2.legend(frameon=False, fontsize=9, loc="lower center")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Avaliação do detector YOLOv8 no meu vídeo", fontsize=15, weight="bold",
                 x=0.012, ha="left", y=0.99)
    fig.text(0.012, 0.015, f"{len(imagens)} frames anotados à mão | casamento por IoU ≥ {args.iou}",
             fontsize=9, color="#6b7280")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    saida_dir = Path("outputs")
    saida_dir.mkdir(parents=True, exist_ok=True)
    caminho_saida = saida_dir / "grafico_avaliacao_detector.png"
    fig.savefig(caminho_saida, dpi=200)
    print(f"\nGrafico salvo: {caminho_saida}")


if __name__ == "__main__":
    main()

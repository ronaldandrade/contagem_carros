
from pathlib import Path

import numpy as np

# Classes do COCO usadas por um modelo pre-treinado.
CLASSES_COCO = {2: "carro", 3: "moto", 5: "onibus", 7: "caminhao"}
# O data.yaml (Roboflow, ordem alfabetica: bus, car, motocycle) numera as classes
# 0/1/2; para avaliar um modelo do COCO e preciso traduzir esses IDs.
ROBOFLOW_PARA_COCO = {0: 5, 1: 2, 2: 3}

# Um modelo treinado neste dataset ja devolve os IDs do data.yaml: nada a traduzir.
CLASSES_CUSTOM = {0: "onibus", 1: "carro", 2: "moto"}
GT_IDENTIDADE = {0: 0, 1: 1, 2: 2}

# O data2 acrescentou uma quarta classe (caminhao) mantendo os IDs 0/1/2 do data1
# -- 100% das caixas do data1 reaparecem inalteradas no data2. Por isso um modelo
# de 3 classes e um de 4 classes podem ser medidos no MESMO gabarito: a linha de
# 'carro' significa a mesma coisa nos dois.
CLASSES_CUSTOM4 = {0: "onibus", 1: "carro", 2: "moto", 3: "caminhao"}
GT_IDENTIDADE4 = {0: 0, 1: 1, 2: 2, 3: 3}


def mapas_de_classe(modo):
    """'coco' para modelo pre-treinado, 'custom'/'custom4' para treinado no dataset."""
    if modo == "custom":
        return CLASSES_CUSTOM, GT_IDENTIDADE
    if modo == "custom4":
        return CLASSES_CUSTOM4, GT_IDENTIDADE4
    return CLASSES_COCO, ROBOFLOW_PARA_COCO


def listar_imagens(diretorio):
    d = Path(diretorio)
    imagens = sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))
    if not imagens:
        raise SystemExit(f"Nenhuma imagem em {diretorio}")
    return imagens


def ler_gt_yolo(caminho_txt, largura, altura, classes, mapa_gt):
    """Le um .txt no formato YOLO e devolve as caixas em pixels, por classe."""
    caixas = {c: [] for c in classes}
    if Path(caminho_txt).exists():
        for linha in Path(caminho_txt).read_text().strip().splitlines():
            if not linha.strip():
                continue
            partes = linha.split()
            cls = mapa_gt.get(int(partes[0]))
            if cls is None or cls not in caixas:
                continue
            cx, cy, w, h = map(float, partes[1:5])
            caixas[cls].append([(cx - w / 2) * largura, (cy - h / 2) * altura,
                                (cx + w / 2) * largura, (cy + h / 2) * altura])
    return {c: np.array(v, dtype=float).reshape(-1, 4) for c, v in caixas.items()}


def detectar(modelo, imagens, classes, conf_min=0.001, imgsz=None):
    """Roda o modelo nas imagens e devolve (preds_por_classe, tamanhos).

    preds_por_classe[c] e uma lista com um array (N, 5) por imagem, cada linha
    [x1, y1, x2, y2, confianca] -- o formato que o nucleo de avaliacao espera.
    """
    preds = {c: [] for c in classes}
    tamanhos = []
    extra = {"imgsz": imgsz} if imgsz else {}
    for img in imagens:
        r = modelo.predict(str(img), conf=conf_min, classes=list(classes),
                           verbose=False, **extra)[0]
        h, w = r.orig_shape
        tamanhos.append((img, w, h))
        caixas = {c: [] for c in classes}
        if r.boxes is not None and len(r.boxes):
            for b, cf, cl in zip(r.boxes.xyxy.cpu().numpy(),
                                 r.boxes.conf.cpu().numpy(),
                                 r.boxes.cls.cpu().numpy().astype(int)):
                if cl in caixas:
                    caixas[cl].append([b[0], b[1], b[2], b[3], cf])
        for c in classes:
            preds[c].append(np.array(caixas[c], dtype=float).reshape(-1, 5))
    return preds, tamanhos


def ler_gts(tamanhos, labels_dir, classes, mapa_gt):
    """Gabarito de cada imagem, na mesma ordem devolvida por detectar()."""
    gts = {c: [] for c in classes}
    for img, w, h in tamanhos:
        gt = ler_gt_yolo(Path(labels_dir) / (img.stem + ".txt"), w, h, classes, mapa_gt)
        for c in classes:
            gts[c].append(gt[c])
    return gts

import numpy as np


def iou_uma_para_muitas(caixa, caixas):
    if len(caixas) == 0:
        return np.zeros(0)
    x1 = np.maximum(caixa[0], caixas[:, 0])
    y1 = np.maximum(caixa[1], caixas[:, 1])
    x2 = np.minimum(caixa[2], caixas[:, 2])
    y2 = np.minimum(caixa[3], caixas[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_c = (caixa[2] - caixa[0]) * (caixa[3] - caixa[1])
    area_s = (caixas[:, 2] - caixas[:, 0]) * (caixas[:, 3] - caixas[:, 1])
    uniao = area_c + area_s - inter
    return np.where(uniao > 0, inter / uniao, 0.0)


def casar_por_imagem(preds, gts, iou_thr):
    ordem = np.argsort(-preds[:, 4]) if len(preds) else np.array([], dtype=int)
    gt_usado = np.zeros(len(gts), dtype=bool)
    marcas = []
    for i in ordem:
        p = preds[i]
        if len(gts) == 0:
            marcas.append((p[4], 0))
            continue
        ious = iou_uma_para_muitas(p[:4], gts)
        ious = np.where(gt_usado, -1.0, ious)
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr:
            gt_usado[j] = True
            marcas.append((p[4], 1))
        else:
            marcas.append((p[4], 0))
    n_gt = len(gts)
    return marcas, n_gt


def curva_precisao_recall(marcas, n_gt_total):
    if n_gt_total == 0:
        return np.array([1.0]), np.array([0.0]), np.array([0.0])
    marcas = sorted(marcas, key=lambda m: -m[0])
    confs = np.array([m[0] for m in marcas])
    tps = np.array([m[1] for m in marcas])
    fps = 1 - tps
    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(fps)
    recall = tp_cum / n_gt_total
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    return precision, recall, confs


def average_precision(precision, recall):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def avaliar_classe(preds_por_img, gts_por_img, iou_thr):
    todas_marcas = []
    n_gt_total = 0
    for preds, gts in zip(preds_por_img, gts_por_img):
        marcas, n_gt = casar_por_imagem(preds, gts, iou_thr)
        todas_marcas.extend(marcas)
        n_gt_total += n_gt
    precision, recall, confs = curva_precisao_recall(todas_marcas, n_gt_total)
    ap = average_precision(precision, recall)
    return {"ap": ap, "precision": precision, "recall": recall,
            "confs": confs, "n_gt": n_gt_total}


def metricas_por_limiar(todas_marcas, n_gt_total, limiares):
    linhas = []
    for c in limiares:
        tp = sum(1 for conf, m in todas_marcas if conf >= c and m == 1)
        fp = sum(1 for conf, m in todas_marcas if conf >= c and m == 0)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / n_gt_total if n_gt_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        linhas.append({"limiar": c, "precisao": precision, "recall": recall, "f1": f1})
    return linhas


def _autoteste():
    gt = np.array([[10, 10, 50, 50], [100, 100, 140, 140]], dtype=float)
    preds = np.array([
        [11, 11, 49, 49, 0.9],
        [101, 101, 139, 139, 0.6],
        [200, 200, 240, 240, 0.3],
    ], dtype=float)
    r = avaliar_classe([preds], [gt], iou_thr=0.5)
    assert abs(r["ap"] - 1.0) < 1e-6, r["ap"]

    preds2 = np.array([
        [11, 11, 49, 49, 0.9],
        [200, 200, 240, 240, 0.8],
        [101, 101, 139, 139, 0.6],
    ], dtype=float)
    r2 = avaliar_classe([preds2], [gt], iou_thr=0.5)
    assert 0.7 < r2["ap"] < 0.9, r2["ap"]

    r3 = avaliar_classe([np.zeros((0, 5))], [gt], iou_thr=0.5)
    assert r3["ap"] == 0.0, r3["ap"]

    print("autoteste OK")
    print(f"  caso ideal (TP,TP,FP):      AP = {r['ap']:.3f}  (esperado 1.000)")
    print(f"  caso com FP no meio:        AP = {r2['ap']:.3f}  (esperado ~0.83)")
    print(f"  caso sem deteccoes:         AP = {r3['ap']:.3f}  (esperado 0.000)")


if __name__ == "__main__":
    _autoteste()

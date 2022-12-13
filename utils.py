from collections import defaultdict
from typing import Dict, List

IOU_THD = 0.6


def intersection(amin, amax, bmin, bmax):
    intmin = max(amin, bmin)
    intmax = min(amax, bmax)
    return max(intmax - intmin, 0)


def iou(a: Dict, b: Dict):  # a, b: xmin, ymin, w, h
    axmin = a["left"]
    axmax = a["left"] + a["width"]
    aymin = a["top"]
    aymax = a["top"] + a["height"]
    aarea = a["width"] * a["height"]

    bxmin = b["left"]
    bxmax = b["left"] + b["width"]
    bymin = b["top"]
    bymax = b["top"] + b["height"]
    barea = b["width"] * b["height"]

    intersection_2d = intersection(axmin, axmax, bxmin, bxmax) * intersection(
        aymin, aymax, bymin, bymax
    )
    union_2d = aarea + barea - intersection_2d
    return intersection_2d / union_2d


def group_by_label(a: List) -> Dict[str, List]:
    result = defaultdict(list)
    for i in a:
        result[i["label"]].append(i)
    return result


def tp_fp_fn(gt, guess, iou_thd=IOU_THD):
    # calculate numbers of true positive and false positive/negative boxes in single label
    tp = 0

    for gt_box in gt:
        for guess_box in guess:
            if iou(gt_box, guess_box) > iou_thd:
                tp += 1
                break

    fp = len(guess) - tp
    fn = len(gt) - tp
    return tp, fp, fn


def fscore(gt: List, guess: List, iou_thd=IOU_THD):
    gt_grouped = group_by_label(gt)
    guess_grouped = group_by_label(guess)
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for label in gt_grouped.keys():
        if label not in guess_grouped:
            total_fn += len(gt_grouped[label])
            continue
        label_tp, label_fp, label_fn = tp_fp_fn(
            gt_grouped[label], guess_grouped[label], iou_thd
        )
        total_tp += label_tp
        total_fp += label_fp
        total_fn += label_fn

    for label in guess_grouped.keys():
        if label not in gt_grouped:
            total_fp += len(guess_grouped[label])

    return (2 * total_tp) / (2 * total_tp + total_fp + total_fn)

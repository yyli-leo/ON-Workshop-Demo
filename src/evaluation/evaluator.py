import re
import numpy as np
from typing import Dict, List


class SpatialOmicsEvaluator:
    """Evaluator for spatial omics results."""

    def __init__(self, classes: List[str] = None):
        if classes is None:
            self.classes = [
                "Proximal tubules", "Distal tubules", "Glomeruli", "Interstitium"
            ]
        else:
            self.classes = [self._norm_label(c) for c in classes]
        self.cls_to_idx = {c: i for i, c in enumerate(self.classes)}

    @staticmethod
    def _norm_label(s: str) -> str:
        """Normalize label name."""
        if s is None:
            return ""
        t = str(s).strip()
        if t == 'Proximial tubules':
            t = 'Proximal tubules'
        return t

    @staticmethod
    def parse_structures(text: str) -> List[str]:
        """Parse predicted structures from model output."""
        if not isinstance(text, str):
            return []

        pattern = re.compile(r'^\s*\d+\.\s*([A-Za-z\s\-_/\[\]\*]+?)\s*:\s*$', re.MULTILINE)
        pattern_inline = re.compile(r'^\s*\d+\.\s*([A-Za-z\s\-_/\[\]\*]+?)\s*:', re.MULTILINE)

        matches = pattern.findall(text)
        if not matches:
            matches = pattern_inline.findall(text)

        seen = set()
        result = []
        for m in matches:
            raw_name = m.strip()
            clean_name = raw_name.replace('[', '').replace(']', '').replace('*', '').strip()
            if clean_name and clean_name.lower() != "other notable features" and clean_name not in seen:
                seen.add(clean_name)
                result.append(clean_name)
        return result

    def compute_metrics(self, y_true: List[str], y_pred: List[List[str]]) -> Dict:
        """Compute accuracy, precision, recall, F1."""
        y_true = [self._norm_label(y) for y in y_true]
        y_pred = [[self._norm_label(p) for p in preds] for preds in y_pred]

        def acc_at_k(k):
            correct = sum(1 for t, preds in zip(y_true, y_pred) if t in preds[:k])
            return correct / len(y_true) if y_true else 0.0

        acc1 = acc_at_k(1)
        acc2 = acc_at_k(2)

        # Confusion matrix
        n = len(self.classes)
        cm = np.zeros((n, n), dtype=int)
        for t, preds in zip(y_true, y_pred):
            if t in self.cls_to_idx and preds and preds[0] in self.cls_to_idx:
                cm[self.cls_to_idx[t]][self.cls_to_idx[preds[0]]] += 1

        top1_metrics = self._calc_prf1(y_true, y_pred, k=1)
        top2_metrics = self._calc_prf1(y_true, y_pred, k=2)

        return {
            'acc@1': acc1,
            'acc@2': acc2,
            'conf_mat': cm,
            'top1': top1_metrics,
            'top2': top2_metrics,
            'classes': self.classes
        }

    def _calc_prf1(self, y_true, y_pred, k=1):
        """Calculate precision, recall, F1."""
        per_cls = {}
        for c in self.classes:
            tp = fp = fn = 0
            for t, preds in zip(y_true, y_pred):
                current_preds = set(preds[:k])
                is_predicted = c in current_preds
                is_true = t == c
                if is_predicted and is_true:
                    tp += 1
                elif is_predicted and not is_true:
                    fp += 1
                elif not is_predicted and is_true:
                    fn += 1

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_cls[c] = {'precision': prec, 'recall': rec, 'f1': f1}

        macro_p = np.mean([v['precision'] for v in per_cls.values()])
        macro_r = np.mean([v['recall'] for v in per_cls.values()])
        macro_f = np.mean([v['f1'] for v in per_cls.values()])

        return {
            'per_class': per_cls,
            'macro': {'precision': macro_p, 'recall': macro_r, 'f1': macro_f}
        }

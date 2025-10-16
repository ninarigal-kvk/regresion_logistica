#!/usr/bin/env python3
"""
kavak_logreg_km.py
------------------
Logistic Regression pipeline (Q1 train / Q2 test) with optional calibration and KM-intent features.

Usage (PowerShell one line):
  python .\\kavak_logreg_km.py --data dataset.csv --outdir outputs_logit_km --topk 0.10 --calibrate isotonic --penalty elasticnet --l1_ratio 0.3 --C 0.8

Expected columns in dataset.csv:
  set, cutoff_date, user_id,
  n_qleads_90d, n_bookings_90d, recency_last_qlead, recency_last_booking,
  n_registers_90d, n_online_schedules_90d, recency_last_register, recency_last_online_schedule,
  total_signals_90d, target_30d
Optional new columns (if you ran the KM SQL patch):
  reg_km_last, reg_km_log1p, reg_km_trailing_zeros,
  reg_km_is_round_1000, reg_km_is_round_500, reg_km_is_round_100, reg_km_is_round_50, reg_km_is_round_10,
  reg_km_mod1000_norm, reg_km_mod100_norm

Outputs (in --outdir):
  - metrics.json
  - metrics_by_segment.json
  - predictions.csv  (set, cutoff_date, user_id, y_true, y_prob, y_pred_topk)
  - coef_importance.csv  (coef, odds_ratio)
  - roc_curve.png, pr_curve.png, calibration_curve.png, lift_curve.png
  - threshold_sweep.csv (1..20%)
  - rec_table.csv (k in {2,5,10,15})
  - monthly_auc.csv / monthly_auc.png
  - cv_warning.txt if CV failed
"""

import json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, brier_score_loss
)

def ensure_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    needed = [
        'set','cutoff_date','user_id',
        'n_qleads_90d','n_bookings_90d','recency_last_qlead','recency_last_booking',
        'n_registers_90d','n_online_schedules_90d','recency_last_register','recency_last_online_schedule',
        'total_signals_90d','target_30d'
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df['cutoff_date'] = pd.to_datetime(df['cutoff_date'])
    df['set'] = df['set'].astype(str)
    df['user_id'] = df['user_id'].astype(str)
    df['target_30d'] = df['target_30d'].astype(int)
    return df

def find_km_features(df):
    extra_num = ['reg_km_log1p','reg_km_trailing_zeros','reg_km_mod1000_norm','reg_km_mod100_norm']
    extra_flags = ['reg_km_is_round_1000','reg_km_is_round_500','reg_km_is_round_100','reg_km_is_round_50','reg_km_is_round_10']
    present_num = [c for c in extra_num if c in df.columns]
    present_flags = [c for c in extra_flags if c in df.columns]
    return present_num, present_flags
    
def build_pipeline(count_cols, recency_cols, extra_num, extra_flags, penalty='l2', l1_ratio=0.5, C=1.0):
    pre = ColumnTransformer(
        transformers=[
            ('cnt_imp', SimpleImputer(strategy='constant', fill_value=0),
             list(count_cols) + list(extra_num) + list(extra_flags)),
            ('rec_imp', SimpleImputer(strategy='constant', fill_value=91),
             list(recency_cols)),
        ],
        remainder='drop'
    )
    if penalty == 'elasticnet':
        clf = LogisticRegression(max_iter=4000, class_weight='balanced', solver='saga',
                                 penalty='elasticnet', l1_ratio=l1_ratio, C=C)
    else:
        clf = LogisticRegression(max_iter=4000, class_weight='balanced', solver='lbfgs',
                                 penalty='l2', C=C)
    pipe = Pipeline([('pre', pre), ('scaler', StandardScaler(with_mean=False)), ('clf', clf)])
    return pipe

def plot_roc(y_true, y_prob, out_png: Path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def plot_pr(y_true, y_prob, out_png: Path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, linewidth=2)
    baseline = y_true.mean()
    plt.hlines(baseline, xmin=0, xmax=1, linestyles='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def plot_calibration(y_true, y_prob, out_png: Path, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='quantile')
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0,1],[0,1], linestyle='--')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def lift_table(y_true, y_prob, q=10) -> pd.DataFrame:
    df = pd.DataFrame({'y': y_true, 'p': y_prob})
    df = df.sort_values('p', ascending=False).reset_index(drop=True)
    df['decile'] = (np.floor(np.arange(len(df)) / (len(df)/q))).astype(int) + 1
    df['decile'] = df['decile'].clip(upper=q)
    tab = df.groupby('decile', as_index=False).agg(n=('y','size'), positives=('y','sum'), mean_p=('p','mean'))
    overall_rate = df['y'].mean()
    tab['response_rate'] = tab['positives'] / tab['n']
    tab['lift'] = tab['response_rate'] / overall_rate if overall_rate > 0 else np.nan
    return tab

def plot_lift(tab: pd.DataFrame, out_png: Path):
    plt.figure()
    plt.plot(tab['decile'], tab['lift'], marker='o')
    plt.xlabel('Decile (1 = most likely)')
    plt.ylabel('Lift')
    plt.title('Lift by Decile')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def coef_importance(model: Pipeline, feature_order) -> pd.DataFrame:
    clf = model.named_steps['clf']
    coefs = clf.coef_.ravel()
    df = pd.DataFrame({'feature': feature_order, 'coef': coefs, 'odds_ratio': np.exp(coefs)}).sort_values('coef', ascending=False)
    return df

def threshold_sweep(y_true, y_prob, max_k=20):
    rows = []
    n = len(y_prob)
    for k in range(1, max_k+1):
        k_frac = k/100.0
        n_top = max(1, int(n * k_frac))
        thr = np.partition(y_prob, -n_top)[-n_top] if n_top < n else np.min(y_prob)
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        rows.append({'k_pct': k, 'k_frac': k_frac, 'threshold': float(thr),
                     'precision': float(precision), 'recall': float(recall),
                     'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(description="Kavak Logistic Regression with KM features")
    ap.add_argument("--data", type=str, required=True, help="Path to dataset.csv")
    ap.add_argument("--outdir", type=str, default="outputs_logit_km", help="Output directory")
    ap.add_argument("--topk", type=float, default=0.1, help="Top-k fraction for thresholding on test (e.g., 0.1 = top 10%)")
    ap.add_argument("--penalty", type=str, default="l2", choices=["l2","elasticnet"], help="Penalty for Logistic Regression")
    ap.add_argument("--l1_ratio", type=float, default=0.5, help="L1 ratio (only if elasticnet)")
    ap.add_argument("--C", type=float, default=1.0, help="Inverse of regularization strength")
    ap.add_argument("--calibrate", type=str, default="none", choices=["none","isotonic","sigmoid"], help="Optional probability calibration on last train month")
    ap.add_argument("--score_data", type=str, default=None,
                help="CSV con nuevos casos a escorear (omite entrenamiento)")
    ap.add_argument("--model", type=str, default=None,
                    help="Ruta a model.joblib (si se omite, usa --outdir/model.joblib)")
    ap.add_argument("--score_out", type=str, default="scores.csv",
                    help="Archivo de salida con probabilidades")

    args = ap.parse_args()
    outdir = Path(args.outdir); ensure_outdir(outdir)

    # === Modo scoring-only ===
    if args.score_data is not None:
        import joblib
        model_path = Path(args.model) if args.model else (Path(args.outdir) / "model.joblib")
        art = joblib.load(model_path)
        predictor = art["predictor"]
        feature_order = art["feature_order"]

        new_df = pd.read_csv(args.score_data, low_memory=False)
        # Asegurar columnas del modelo (crea NaN si faltan)
        for col in feature_order:
            if col not in new_df.columns:
                new_df[col] = np.nan

        X_new = new_df[feature_order]
        p_new = predictor.predict_proba(X_new)[:, 1]
        out_scores = Path(args.outdir) / args.score_out
        pd.DataFrame({
            "user_id": new_df["user_id"] if "user_id" in new_df.columns else range(len(new_df)),
            "y_prob": p_new
        }).to_csv(out_scores, index=False)
        print(json.dumps({"ok": True, "mode": "score", "scored": int(len(p_new)), "scores": str(out_scores)}, indent=2))
        return

    df = load_data(Path(args.data))
    train_df = df[df['set'].str.lower() == 'train'].copy()
    test_df  = df[df['set'].str.lower() == 'test'].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Train or Test split is empty. Check 'set' column values.")

    # Features
    count_cols = ['n_qleads_90d','n_bookings_90d','n_registers_90d','n_online_schedules_90d','total_signals_90d']
    recency_cols = ['recency_last_qlead','recency_last_booking','recency_last_register','recency_last_online_schedule']

    extra_num, extra_flags = find_km_features(df)  # may be empty lists if not present
    feature_order = count_cols + recency_cols + extra_num + extra_flags

    # Build and fit
    pipe = build_pipeline(count_cols, recency_cols, extra_num, extra_flags,
                          penalty=args.penalty, l1_ratio=args.l1_ratio, C=args.C)

    # Temporal validation: last train month
    """
    last_train_cut = train_df['cutoff_date'].max()
    fit_df = train_df[train_df['cutoff_date'] < last_train_cut]
    val_df = train_df[train_df['cutoff_date'] == last_train_cut]
    if fit_df.empty or val_df.empty:
        fit_df = train_df.sample(frac=0.8, random_state=42)
        val_df = train_df.drop(fit_df.index)
    """

    h = pd.util.hash_pandas_object(train_df['user_id'], index=False) % 100
    fit_df = train_df[h < 80]
    val_df = train_df[h >= 80]

    X_fit = fit_df[feature_order]; y_fit = fit_df['target_30d'].values
    X_val = val_df[feature_order]; y_val = val_df['target_30d'].values

    pipe.fit(X_fit, y_fit)

    # Optional calibration wrapper
    predictor = pipe
    if args.calibrate != 'none':
        method = 'isotonic' if args.calibrate=='isotonic' else 'sigmoid'
        cal = CalibratedClassifierCV(predictor, method=method, cv='prefit')
        cal.fit(X_val, y_val)
        predictor = cal

    # === Guardar modelo entrenado para scoring futuro ===
    import joblib
    joblib.dump({"predictor": predictor, "feature_order": feature_order},
                Path(outdir) / "model.joblib")

    # Predict
    X_test = test_df[feature_order]
    y_test = test_df['target_30d'].values
    p_test = predictor.predict_proba(X_test)[:,1]

    # === Learning curve (AUC y PR-AUC) en VALIDACIÓN fija (20% por user_id) ===
    def plot_learning_curve_fixed_val(pipe_builder, full_df, feature_order, outdir):
        import numpy as np, matplotlib.pyplot as plt
        from sklearn.metrics import roc_auc_score, average_precision_score

        # split determinístico 80/20 por user_id sobre TRAIN
        train_df = full_df[full_df['set'].str.lower()=='train'].copy()
        h = pd.util.hash_pandas_object(train_df['user_id'], index=False) % 100
        fit_df = train_df[h < 80].copy()
        val_df = train_df[h >= 80].copy()

        X_val = val_df[feature_order]; y_val = val_df['target_30d'].values

        fracs = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        aucs, praucs, sizes = [], [], []

        for f in fracs:
            sub = fit_df.sample(frac=f, random_state=42)
            X_fit = sub[feature_order]; y_fit = sub['target_30d'].values
            pipe = pipe_builder()
            pipe.fit(X_fit, y_fit)
            p = pipe.predict_proba(X_val)[:,1]
            sizes.append(len(sub))
            aucs.append(roc_auc_score(y_val, p))
            praucs.append(average_precision_score(y_val, p))

        # AUC
        plt.figure()
        plt.plot(sizes, aucs, marker='o')
        plt.xlabel('Tamaño de entrenamiento'); plt.ylabel('AUC (validación)')
        plt.title('Learning curve - AUC'); plt.tight_layout()
        plt.savefig(Path(outdir)/'learning_curve_auc.png'); plt.close()

        # PR-AUC
        plt.figure()
        plt.plot(sizes, praucs, marker='o')
        plt.xlabel('Tamaño de entrenamiento'); plt.ylabel('PR-AUC (validación)')
        plt.title('Learning curve - PR-AUC'); plt.tight_layout()
        plt.savefig(Path(outdir)/'learning_curve_prauc.png'); plt.close()

    # Llamado, reutilizando tu configuración actual
    def _builder():
        return build_pipeline(count_cols, recency_cols, extra_num, extra_flags,
                            penalty=args.penalty, l1_ratio=args.l1_ratio, C=args.C)
    plot_learning_curve_fixed_val(_builder, df, feature_order, outdir)


    # Metrics
    metrics = {
        'train': {
            'roc_auc': float(roc_auc_score(train_df['target_30d'].values,
                                           predictor.predict_proba(df[df['set'].str.lower()=='train'][feature_order])[:,1])),
            'pr_auc': float(average_precision_score(train_df['target_30d'].values,
                                                    predictor.predict_proba(df[df['set'].str.lower()=='train'][feature_order])[:,1])),
            'brier': float(brier_score_loss(train_df['target_30d'].values,
                                            predictor.predict_proba(df[df['set'].str.lower()=='train'][feature_order])[:,1])),
            'positive_rate': float(np.mean(train_df['target_30d'].values))
        },
        'test': {
            'roc_auc': float(roc_auc_score(y_test, p_test)),
            'pr_auc': float(average_precision_score(y_test, p_test)),
            'brier': float(brier_score_loss(y_test, p_test)),
            'positive_rate': float(np.mean(y_test))
        },
        'features_used': feature_order
    }

    # Threshold by top-k on TEST
    topk = args.topk
    n_top = max(1, int(len(p_test) * topk))
    thr = np.partition(p_test, -n_top)[-n_top] if n_top < len(p_test) else np.min(p_test)
    y_pred = (p_test >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
    metrics['test_thresholding'] = {
        'topk_fraction': topk,
        'threshold': float(thr),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'precision_at_k': float(tp / (tp + fp)) if (tp + fp) > 0 else None,
        'recall_at_k': float(tp / (tp + fn)) if (tp + fn) > 0 else None
    }

    # Save metrics
    with open(outdir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Predictions file
    pred_df = test_df[['set','cutoff_date','user_id']].copy()
    pred_df['y_true'] = y_test
    pred_df['y_prob'] = p_test
    pred_df['y_pred_topk'] = y_pred
    pred_df.to_csv(outdir / 'predictions.csv', index=False)

    # Coefficients / importances (only if not calibrated wrapper)
    # base_model = pipe if args.calibrate=='none' else predictor.base_estimator
    base_model = pipe
    coef_df = coef_importance(base_model, feature_order)
    coef_df.to_csv(outdir / 'coef_importance.csv', index=False)

    # Plots
    plot_roc(y_test, p_test, outdir / 'roc_curve.png')
    plot_pr(y_test, p_test, outdir / 'pr_curve.png')
    plot_calibration(y_test, p_test, outdir / 'calibration_curve.png')
    lift = lift_table(y_test, p_test, q=10)
    lift.to_csv(outdir / 'lift_table.csv', index=False)
    plot_lift(lift, outdir / 'lift_curve.png')

    # Threshold sweep 1..20%
    sweep = threshold_sweep(y_test, p_test, max_k=20)
    sweep.to_csv(outdir / 'threshold_sweep.csv', index=False)
    rec_table = sweep[sweep['k_pct'].isin([2,5,10,15])]
    rec_table.to_csv(outdir / 'rec_table.csv', index=False)

    # Monthly AUC on test
    auc_by_set = (
    test_df.assign(p=p_test)
           .groupby('set')
           .apply(lambda g: roc_auc_score(g['target_30d'], g['p']) if g['target_30d'].nunique()>1 else np.nan)
           .reset_index(name='auc')
    )
    auc_by_set.to_csv(outdir / 'auc_by_set.csv', index=False)

    print(json.dumps({'ok': True, 'outputs': str(outdir)}, indent=2))

    

if __name__ == "__main__":
    main()

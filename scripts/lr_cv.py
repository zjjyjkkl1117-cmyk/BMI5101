"""BMI5101 P15 — 逻辑回归（LR），按全组统一的划分和 5 折做交叉验证

和蔡嘉怡的 xgboost_cv.py 对齐：
  - 划分：splits/ 里的 A_train / A_test / B_train / B_test
    （由 create_train_test_split.py 生成）
  - 5 折：splits/cv_folds.csv，每个病人固定属于哪一折；文件不存在时按
    xgboost_cv.py 的同一段代码生成，存在时核对一致
  - 每一折只用训练折拟合：预处理（中位数补缺失 + 缺失指示 + 标准化，
    类别变量独热编码）→ LASSO（L1 逻辑回归，C = 0.5）选变量 → LR
    前两步和 xgboost_cv.py 完全相同，只把最后的 XGBoost 换成 LR
  - LR 的正则化强度 C：在每个训练折内部再做 5 折交叉验证来选（嵌套），
    验证折不参与调参
  - 打分：组长的 metrics.py（Youden 阈值只用训练集 5 折 out-of-fold 概率来定）
    只有 --final-test 需要 metrics.py，放在同一个 scripts/ 文件夹里即可

用法（在仓库根目录，先运行 create_train_test_split.py 生成 splits/）：
    python scripts/lr_cv.py                # 只做交叉验证，不碰测试集
    python scripts/lr_cv.py --final-test   # 做法定好之后，测试集只评估一次
结果写到 results/lr/（不要上传）
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore", category=FutureWarning)
# sklearn 1.8+ 提示 penalty 参数将改名；实测 penalty="l1" 仍按 L1 生效，为和 xgboost_cv.py 一致保留原写法
warnings.filterwarnings("ignore", message="Inconsistent values: penalty=l1")

SEED = 42
TARGET = "died_in_hospital"
ID = "subject_id"
FOLDS = 5
C_GRID = np.logspace(-3, 2, 11)  # LR 的 C 候选值：0.001 … 100
# 和 xgboost_cv.py 相同：编号、其他结局、事后标记都不能当预测变量
DROP_FEATURES = {
    "subject_id", "hadm_id", "first_stay_id",
    "died_in_hospital", "died_28d", "died_in_icu", "died_or_hospice",
    "paper_would_exclude",
}


def load_data(data_dir: Path, name: str):
    tr = pd.read_csv(data_dir / f"{name}_train.csv")
    te = pd.read_csv(data_dir / f"{name}_test.csv")
    for label, df in [("train", tr), ("test", te)]:
        if df[ID].isna().any() or not df[ID].is_unique:
            raise ValueError(f"{name}_{label}: subject_id 有缺失或重复")
        if not df[TARGET].isin([0, 1]).all():
            raise ValueError(f"{name}_{label}: 结局变量有缺失或不是 0/1")
    if set(tr[ID]) & set(te[ID]):
        raise ValueError(f"{name}: 有病人同时出现在训练集和测试集")
    return tr.sort_values(ID).reset_index(drop=True), te.sort_values(ID).reset_index(drop=True)


def get_folds(a_train: pd.DataFrame, fold_file: Path) -> np.ndarray:
    """和 xgboost_cv.py 同一段代码算 5 折；已有 cv_folds.csv 时必须一致。"""
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    folds = np.full(len(a_train), -1, dtype=int)
    for i, (_, valid_idx) in enumerate(cv.split(a_train, a_train[TARGET])):
        folds[valid_idx] = i
    new = pd.Series(folds, index=a_train[ID].to_numpy())
    if fold_file.exists():
        old = pd.read_csv(fold_file).set_index(ID)["fold"]
        if not old.sort_index().equals(new.sort_index()):
            raise ValueError("已有的 cv_folds.csv 和按同样代码算出的不一致，请先在组里核对")
        print(f"[折] 使用已有的 {fold_file.name}，与 xgboost_cv.py 的算法一致")
        return old.reindex(a_train[ID]).to_numpy()
    pd.DataFrame({ID: a_train[ID], "fold": folds}).to_csv(fold_file, index=False)
    print(f"[折] 已按 xgboost_cv.py 的算法生成 {fold_file}")
    return folds


def build_pipeline(x: pd.DataFrame) -> Pipeline:
    numeric = x.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in x if c not in numeric]
    prep = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True,
                                     add_indicator=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ], sparse_threshold=0)
    return Pipeline([
        ("preprocess", prep),
        ("lasso", SelectFromModel(
            LogisticRegression(penalty="l1", solver="liblinear", C=0.5,
                               max_iter=3000, random_state=SEED),
            threshold=1e-8,
        )),
        ("lr", LogisticRegressionCV(
            Cs=C_GRID, cv=StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED),
            scoring="roc_auc", max_iter=5000, use_legacy_attributes=False,
        )),
    ])


def selected_names(pipe: Pipeline) -> np.ndarray:
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    return names[pipe.named_steps["lasso"].get_support()]


def chosen_c(pipe: Pipeline) -> float:
    return float(np.ravel(pipe.named_steps["lr"].C_)[0])


def main():
    project = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=project / "splits")
    p.add_argument("--output-dir", type=Path, default=project / "results" / "lr")
    p.add_argument("--final-test", action="store_true",
                   help="用全部训练集拟合，测试集只评估一次")
    args = p.parse_args()
    if args.final_test:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            from metrics import evaluate_model, format_table  # 组长的评估标准
        except ImportError:
            sys.exit("--final-test 需要组长的 metrics.py，请把它放进 scripts/ 文件夹")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    a_train, a_test = load_data(args.data_dir, "A")
    b_train, b_test = load_data(args.data_dir, "B")
    for part, a, b in [("train", a_train, b_train), ("test", a_test, b_test)]:
        if not a[ID].equals(b[ID]) or not a[TARGET].equals(b[TARGET]):
            raise ValueError(f"Model A/B 的 {part} 病人或结局不一致")
    folds = get_folds(a_train, args.data_dir / "cv_folds.csv")

    fold_rows, feat_rows, oof_rows, tables, summary = [], [], [], [], {}
    for name, tr, te in [("A", a_train, a_test), ("B", b_train, b_test)]:
        cols = [c for c in tr.columns if c not in DROP_FEATURES]
        x, y = tr[cols], tr[TARGET].to_numpy()
        print(f"\nModel {name}: 训练 {len(tr)} 人，测试 {len(te)} 人，原始变量 {len(cols)} 个")

        oof = np.full(len(tr), np.nan)
        for fold in range(FOLDS):
            ti, vi = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            pipe = build_pipeline(x).fit(x.iloc[ti], y[ti])
            oof[vi] = pipe.predict_proba(x.iloc[vi])[:, 1]
            chosen = selected_names(pipe)
            fold_rows.append({"model": name, "fold": fold, "n_train": len(ti),
                              "n_validation": len(vi), "selected_features": len(chosen),
                              "lr_C": chosen_c(pipe),
                              "validation_auroc": roc_auc_score(y[vi], oof[vi])})
            feat_rows.extend({"model": name, "fold": fold, "feature": f} for f in chosen)
            print(f"  第 {fold} 折：AUROC={fold_rows[-1]['validation_auroc']:.3f}，"
                  f"LASSO 选出 {len(chosen)} 个变量，LR 的 C={chosen_c(pipe):.4g}")
        oof_rows.append(pd.DataFrame({"model": name, ID: tr[ID], "fold": folds,
                                      "observed": y, "oof_probability": oof}))
        summary[name] = {"n_raw_predictors": len(cols),
                         "cv_pooled_auroc": float(roc_auc_score(y, oof))}

        if args.final_test:
            final = build_pipeline(x).fit(x, y)
            p_tr = final.predict_proba(x)[:, 1]
            p_te = final.predict_proba(te[cols])[:, 1]
            table = evaluate_model(f"LR_{name}", y, p_tr, te[TARGET].to_numpy(), p_te, oof)
            table["Cohort"] = table["Cohort"].replace({"Train (10-fold CV)": "Train (5-fold CV)"})
            tables.append(table)
            names = selected_names(final)
            pd.DataFrame({"feature": names,
                          "coefficient": final.named_steps["lr"].coef_.ravel()}) \
                .sort_values("coefficient", key=np.abs, ascending=False) \
                .to_csv(out / f"final_coefficients_{name}.csv", index=False)
            pd.DataFrame({ID: te[ID], "observed": te[TARGET],
                          "predicted_probability": p_te}) \
                .to_csv(out / f"test_predictions_{name}.csv", index=False)
            summary[name].update({"final_n_selected": len(names), "final_lr_C": chosen_c(final)})
            print(f"  最终模型：LASSO 选出 {len(names)} 个变量，LR 的 C={chosen_c(final):.4g}")

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(out / "cv_fold_metrics.csv", index=False)
    feats = pd.DataFrame(feat_rows)
    feats.to_csv(out / "cv_selected_features.csv", index=False)
    feats.groupby(["model", "feature"]).size().rename("folds_selected").reset_index() \
        .sort_values(["model", "folds_selected", "feature"], ascending=[True, False, True]) \
        .to_csv(out / "cv_feature_selection_frequency.csv", index=False)
    pd.concat(oof_rows).to_csv(out / "cv_oof_predictions.csv", index=False)

    print("\n5 折交叉验证（每折 AUROC 的平均 / 标准差）：")
    print(folds_df.groupby("model")[["validation_auroc", "selected_features"]]
          .agg(["mean", "std"]).round(3).to_string())

    if args.final_test:
        perf = pd.concat(tables, ignore_index=True)
        perf.to_csv(out / "lr_performance.csv", index=False)
        print("\n全组标准评估（metrics.py）：")
        print(format_table(perf).to_string(index=False))
    else:
        print("\n测试集没有动。做法定好之后再加 --final-test 运行一次。")

    (out / "run_summary.json").write_text(json.dumps({
        "split_dir": str(args.data_dir), "folds": FOLDS, "seed": SEED,
        "lasso": "L1 LogisticRegression(liblinear, C=0.5), same as xgboost_cv.py",
        "lr": "LogisticRegressionCV(L2, inner 5-fold, scoring=roc_auc)",
        "C_grid": C_GRID.tolist(), "final_test_run": args.final_test, "models": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果保存在 {out}")


if __name__ == "__main__":
    main()

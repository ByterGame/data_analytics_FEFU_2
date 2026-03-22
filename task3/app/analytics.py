import pandas as pd
import numpy as np


def compute_statistics(df: pd.DataFrame) -> dict:
    """Вычисляет сводную статистику по DataFrame."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    id_cols = [c for c in numeric_cols if c.lower() in ("id", "index", "unnamed: 0")]
    numeric_cols = [c for c in numeric_cols if c not in id_cols]

    stats = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {
            col: int(count)
            for col, count in df.isnull().sum().items()
            if count > 0
        },
        "total_missing": int(df.isnull().sum().sum()),
        "numeric_stats": {},
        "categorical_stats": {},
    }

    for col in numeric_cols:
        s = df[col].dropna()
        stats["numeric_stats"][col] = {
            "mean": round(float(s.mean()), 2),
            "median": round(float(s.median()), 2),
            "std": round(float(s.std()), 2),
            "min": round(float(s.min()), 2),
            "max": round(float(s.max()), 2),
            "q25": round(float(s.quantile(0.25)), 2),
            "q75": round(float(s.quantile(0.75)), 2),
        }

    for col in categorical_cols:
        vc = df[col].value_counts().head(10)
        stats["categorical_stats"][col] = {
            str(k): int(v) for k, v in vc.items()
        }

    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                val = corr.iloc[i, j]
                if not np.isnan(val):
                    pairs.append({
                        "col1": numeric_cols[i],
                        "col2": numeric_cols[j],
                        "correlation": round(float(val), 3),
                    })
        pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        stats["top_correlations"] = pairs[:10]

    return stats

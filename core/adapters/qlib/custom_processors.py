import logging
import numpy as np
import pandas as pd
from qlib.data.dataset.processor import Processor, get_group_columns

logger = logging.getLogger(__name__)


class CSFeatureAugment(Processor):
    """为每个特征追加截面百分位排名 (_cspct)，可选截面 z-score (_csz)"""

    def __init__(self, fields_group="feature", add_rank=True, add_zscore=False, top_n=None):
        self.fields_group = fields_group
        self.add_rank = add_rank
        self.add_zscore = add_zscore
        self.top_n = top_n

    def __call__(self, df):
        cols = list(get_group_columns(df, self.fields_group))
        if not cols:
            return df

        new_cols = {}
        for col in cols:
            series = df[col]
            base_name = col if isinstance(col, str) else col[1]
            group = col[0] if isinstance(col, tuple) else self.fields_group

            if self.add_rank:
                new_cols[(group, f"{base_name}_cspct")] = series.groupby(level="datetime", sort=False).rank(pct=True)

            if self.add_zscore:
                new_cols[(group, f"{base_name}_csz")] = series.groupby(level="datetime", sort=False).transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-12)
                )

        if new_cols:
            new_df = pd.DataFrame(new_cols, index=df.index)
            result = pd.concat([df, new_df], axis=1)
            result = result.infer_objects()
            return result

        return df

    def _keep_top_variance(self, df, n):
        cols = [c for c in get_group_columns(df, self.fields_group)
                if isinstance(c, tuple) and (str(c[1]).endswith("_cspct") or str(c[1]).endswith("_csz"))]
        if len(cols) <= n:
            return
        variances = df[cols].var()
        keep = set(variances.nlargest(n).index)
        for c in cols:
            if c not in keep:
                df.drop(columns=[c], inplace=True)

    def readonly(self):
        return False

    def is_for_infer(self):
        return True


class FactorCrossover(Processor):
    """取方差最大的前 max_features 个特征，两两相乘生成交互特征 (_x_)"""

    def __init__(self, fields_group="feature", max_features=15, mode="product"):
        self.fields_group = fields_group
        self.max_features = max_features
        self.mode = mode

    def __call__(self, df):
        all_cols = list(get_group_columns(df, self.fields_group))
        if len(all_cols) < 2:
            return df

        selected = self._select_by_variance(df, all_cols, self.max_features)

        new_cols = {}
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                ci, cj = selected[i], selected[j]
                ni = ci[1] if isinstance(ci, tuple) else ci
                nj = cj[1] if isinstance(cj, tuple) else cj
                group = ci[0] if isinstance(ci, tuple) else self.fields_group

                if self.mode in ("product", "both"):
                    new_cols[(group, f"{ni}_x_{nj}")] = df[ci] * df[cj]
                if self.mode in ("ratio", "both"):
                    new_cols[(group, f"{ni}_div_{nj}")] = df[ci] / (df[cj] + 1e-12)

        if new_cols:
            new_df = pd.DataFrame(new_cols, index=df.index)
            return pd.concat([df, new_df], axis=1)

        return df

    def _select_by_variance(self, df, cols, n):
        if len(cols) <= n:
            return cols
        variances = df[cols].var()
        top_idx = variances.argsort()[-n:]
        return [cols[i] for i in top_idx]

    def readonly(self):
        return False

    def is_for_infer(self):
        return True


class CSWinsorize(Processor):
    """逐日截面 winsorize (5%/95%) 抗极值 — 对标 fundamental._compute_zscores"""

    def __init__(self, fields_group="feature", lower=0.05, upper=0.95):
        self.fields_group = fields_group
        self.lower = lower
        self.upper = upper

    def __call__(self, df):
        cols = list(get_group_columns(df, self.fields_group))
        if not cols:
            return df
        for col in cols:
            df[col] = df[col].groupby(level="datetime", sort=False).transform(
                self._winsorize_group
            )
        return df

    def _winsorize_group(self, g):
        if g.notna().sum() < 10:
            return g
        lo, hi = g.quantile(self.lower), g.quantile(self.upper)
        if hi <= lo:
            return g
        return g.clip(lo, hi)

    def readonly(self):
        return False

    def is_for_infer(self):
        return True

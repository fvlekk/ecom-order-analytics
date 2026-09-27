"""Leak-free rolling history: what was already known about a group on the day an order came in."""

import numpy as np
import pandas as pd


def prior_window_sums(
    df: pd.DataFrame,
    keys: list[str],
    query_time: str,
    event_time: str,
    values: list[str],
    window: str = "60D",
    known: pd.Series | None = None,
) -> pd.DataFrame:
    """For each row, sums of `values` over events of the same `keys` whose `event_time` falls in
    the `window` strictly before the row's `query_time` day (same-day events are excluded).
    `known` masks which rows are events at all (e.g. orders with a final outcome).
    Returns columns n (event count) and one column per value, aligned on df.index.
    """
    mask = known if known is not None else pd.Series(True, index=df.index)
    ev = df.loc[mask].dropna(subset=[*keys, event_time]).copy()
    empty = pd.DataFrame({"n": 0.0, **{v: np.nan for v in values}}, index=df.index)
    if ev.empty:
        return empty
    ev["day"] = ev[event_time].dt.normalize()
    ev["n"] = 1
    daily = ev.groupby([*keys, "day"])[["n", *values]].sum()

    # One row per key and calendar day; closed="left" keeps day D's own events out of day D.
    last = df[query_time].max().normalize()
    days = pd.date_range(
        daily.index.get_level_values("day").min(), max(last, daily.index.get_level_values("day").max()), name="day"
    )
    parts = []
    for key_values, g in daily.groupby(level=keys):
        key_values = key_values if isinstance(key_values, tuple) else (key_values,)
        r = g.droplevel(keys).reindex(days, fill_value=0).rolling(window, closed="left").sum()
        parts.append(r.assign(**dict(zip(keys, key_values, strict=True))))
    rolled = pd.concat(parts).reset_index()

    left = df[keys].assign(day=df[query_time].dt.normalize(), _row=np.arange(len(df)))
    m = left.merge(rolled, on=[*keys, "day"], how="left").sort_values("_row")
    out = m[["n", *values]].set_axis(df.index)
    out["n"] = out["n"].fillna(0)
    return out

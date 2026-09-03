import pandas as pd
import holidays

# Each history feature can be sourced two ways: from the raw per-day column in
# tourism_data.csv, or from the already-aggregated column in processed_data.csv.
# Both resolve to the same number, because a (site, month, day) slice selects the
# same rows in either file and every row in that slice carries the same aggregate.
HIST_COLUMNS = {
    'hist_avg_temp': ('temp', 'hist_avg_temp'),
    'hist_avg_precip': ('precip', 'hist_avg_precip'),
    'hist_event_count': ('event_count', 'hist_event_count'),
    'hist_trend_score': ('trend_score', 'hist_trend_score'),
}

def load_tourism_data(path):
    df =pd.read_csv(path, parse_dates=['date'])
    return df

def _hist_mean(hist, raw_col, agg_col):
    if raw_col in hist.columns:
        return hist[raw_col].mean()
    if agg_col in hist.columns:
        return hist[agg_col].mean()
    raise KeyError(
        f"history frame has neither '{raw_col}' nor '{agg_col}'; defaulting it to 0 "
        "would feed the model a feature it never saw during training"
    )

def feature_engineer_date(full_df, site, date):
    out = {}
    out['month'] =date.month
    out['weekday'] =date.weekday()
    out['is_holiday'] = int(date in holidays.India(years=[date.year]))
    out['site'] = site

    hist = full_df[
        (full_df['site'] == site) &
        (full_df['date'].dt.month == date.month) &
        (full_df['date'].dt.day == date.day)
        ]

    out['hist_avg_tourists'] = hist['tourists'].mean() if not hist.empty else 0
    for name, (raw_col, agg_col) in HIST_COLUMNS.items():
        out[name] = _hist_mean(hist, raw_col, agg_col)
    return pd.DataFrame([out])

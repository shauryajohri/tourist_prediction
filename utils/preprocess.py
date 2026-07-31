import pandas as pd
import holidays

def load_tourism_data(path):
    df =pd.read_csv(path, parse_dates=['date'])
    return df

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
    out['hist_avg_temp'] = hist['temp'].mean() if 'temp' in hist.columns else 0
    out['hist_avg_precip'] = hist['precip'].mean() if 'precip' in hist.columns else 0
    out['hist_event_count'] = hist['event_count'].mean() if 'event_count' in hist.columns else 0
    out['hist_trend_score'] = hist['trend_score'].mean() if 'trend_score' in hist.columns else 0
    return pd.DataFrame([out])


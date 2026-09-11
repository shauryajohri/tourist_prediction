"""Precompute the model's answers for the website and write shaant/assets/data.js.

The model is a Python XGBoost classifier, which cannot run in a browser. So this
script asks it every question the website can ask, ahead of time: every site on
every day in a date range. The answers are packed into short strings and saved
as a JavaScript file the page loads directly, so the site needs no server.

Run it after retraining (python train_model.py), from this folder:

    python export_site_data.py
    python export_site_data.py --start 2027-01-01 --end 2029-12-31

Output format, per site in window.PRED.sites:
    c  one letter per day from `start`: M (Mild), S (Suitable), O (Overcrowded)
    p  two digits per day: the model's confidence in that answer, 0-99 percent
    v  three digits per calendar day, slot (month-1)*31 + (day-1): the average
       visitors on that calendar day in the history (the model's main input)
    m  twelve numbers: average visitors per month, January first
"""
import argparse
import json
import os
from datetime import date, timedelta

import holidays
import joblib
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('--start', default='2026-09-01', help='first date to predict (YYYY-MM-DD)')
parser.add_argument('--end', default='2028-12-31', help='last date to predict (YYYY-MM-DD)')
parser.add_argument('--out', default='shaant/assets/data.js', help='where to write the JavaScript file')
args = parser.parse_args()

model = joblib.load('models/tourism_classifier.pkl')
site_enc = joblib.load('models/site_encoder.pkl')
targ_enc = joblib.load('models/target_encoder.pkl')

raw = pd.read_csv('data/tourism_data.csv', parse_dates=['date'])
raw['m'] = raw.date.dt.month
raw['d'] = raw.date.dt.day

# The same history features train_model.py builds through feature_engineer_date:
# the mean of each column over every past year, for one site on one calendar day.
g = raw.groupby(['site', 'm', 'd']).agg(
    hist_avg_tourists=('tourists', 'mean'), hist_avg_temp=('temp', 'mean'),
    hist_avg_precip=('precip', 'mean'), hist_event_count=('event_count', 'mean'),
    hist_trend_score=('trend_score', 'mean')).reset_index()
# Monthly means, used only where a calendar day has no history (29 February).
gm = raw.groupby(['site', 'm']).agg(
    t=('tourists', 'mean'), tp=('temp', 'mean'), pr=('precip', 'mean'),
    ec=('event_count', 'mean'), ts=('trend_score', 'mean')).reset_index()

start = date.fromisoformat(args.start)
end = date.fromisoformat(args.end)
ndays = (end - start).days + 1
dates = [start + timedelta(days=i) for i in range(ndays)]
hol = set()
for y in range(start.year, end.year + 1):
    hol |= set(holidays.India(years=[y]).keys())

print(f"Predicting {len(site_enc.classes_)} sites x {ndays} days ({start} to {end})...")
sites = list(site_enc.classes_)
rows = [(s, dt, dt.month, dt.weekday(), int(dt in hol), dt.day) for s in sites for dt in dates]
grid = pd.DataFrame(rows, columns=['site', 'dt', 'month', 'weekday', 'is_holiday', 'd'])
grid = grid.merge(g, left_on=['site', 'month', 'd'], right_on=['site', 'm', 'd'], how='left')
grid = grid.merge(gm, left_on=['site', 'month'], right_on=['site', 'm'], how='left', suffixes=('', '_mm'))
for c, f in [('hist_avg_tourists', 't'), ('hist_avg_temp', 'tp'), ('hist_avg_precip', 'pr'),
             ('hist_event_count', 'ec'), ('hist_trend_score', 'ts')]:
    grid[c] = grid[c].fillna(grid[f])

X = pd.DataFrame({
    'month': grid.month, 'weekday': grid.weekday, 'is_holiday': grid.is_holiday,
    'site': site_enc.transform(grid.site),
    'hist_avg_tourists': grid.hist_avg_tourists, 'hist_avg_temp': grid.hist_avg_temp,
    'hist_avg_precip': grid.hist_avg_precip, 'hist_event_count': grid.hist_event_count,
    'hist_trend_score': grid.hist_trend_score,
})[list(model.feature_names_in_)]

proba = model.predict_proba(X)
grid['cond'] = targ_enc.inverse_transform(proba.argmax(axis=1))
grid['conf'] = proba.max(axis=1)

CODE = {'Mild': 'M', 'Suitable': 'S', 'Overcrowded': 'O'}
out = {}
for s in sites:
    sub = grid[grid.site == s].sort_values('dt')
    vslots = ['000'] * 372
    for _, r in g[g.site == s].iterrows():
        vslots[int(r.m - 1) * 31 + int(r.d - 1)] = str(min(999, int(round(r.hist_avg_tourists)))).zfill(3)
    gms = gm[gm.site == s].set_index('m')
    for mi in range(1, 13):                     # slots with no calendar day (30 Feb and so on) get the month mean
        mv = str(min(999, int(round(gms.loc[mi, 't'])))).zfill(3)
        for di in range(31):
            if vslots[(mi - 1) * 31 + di] == '000':
                vslots[(mi - 1) * 31 + di] = mv
    out[s] = {
        'c': ''.join(CODE[c] for c in sub.cond),
        'p': ''.join(str(min(99, int(round(c * 100)))).zfill(2) for c in sub.conf),
        'v': ''.join(vslots),
        'm': [int(round(x)) for x in gms['t'].reindex(range(1, 13)).tolist()],
    }

payload = {'start': start.isoformat(), 'days': ndays, 'sites': out}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, 'w', newline='\n') as fh:
    fh.write('window.PRED=' + json.dumps(payload, separators=(',', ':')) + ';\n')

print(f"Wrote {args.out} ({round(os.path.getsize(args.out) / 1024)} KB)")
print("Mix of answers:", grid.cond.value_counts(normalize=True).round(3).to_dict())

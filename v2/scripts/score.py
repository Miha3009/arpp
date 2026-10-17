import base64, io, json, argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from dateutil.relativedelta import relativedelta
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

p = argparse.ArgumentParser()
p.add_argument('--forecasts', required=True)
p.add_argument('--region', required=True)
p.add_argument('--type', required=True, choices=['week', 'month'])
p.add_argument('--maps', action='store_true', help='рисовать карты (страница 1)')
p.add_argument('--out', default='comparison.html')
args = p.parse_args()

forecasts = [s.strip() for s in args.forecasts.split(',') if s.strip()]
min_lat, max_lat, min_lon, max_lon = map(float, args.region.split(','))
type_ = args.type

BASE = Path('../data/score/swe')
SCALE_ANOM   = (-50, 50)
SCALE_ABS    = (0, 150)
SCALE_RMSE   = (0, 50)
SCALE_ACC    = (-1, 1)
SCALE_DIFF = {'rmse': 25, 'acc': 1}
LEADS_WEEK   = 17
LEADS_MONTH  = 4

VAR      = 'swe_anom_weekly' if type_ == 'week' else 'swe_anom_monthly'
TIME_VAR = 'time_weekly'     if type_ == 'week' else 'time_monthly'
LEADS    = LEADS_WEEK        if type_ == 'week' else LEADS_MONTH
CLIM_VAR = 'swe_clim_weekly' if type_ == 'week' else 'swe_clim_monthly'

def interp_circular_lon(da, target_lat, target_lon, lon_period=360.0):
    lon = da['lon'].values
    lon_norm = np.mod(lon, lon_period)
    da = da.assign_coords(lon=lon_norm).sortby('lon')
    wrap = da.isel(lon=slice(0, 1)).assign_coords(lon=[lon_norm[0] + lon_period])
    da = xr.concat([da, wrap], dim='lon')
    target_lon_norm = np.mod(target_lon, lon_period)
    return da.interp(lat=target_lat, lon=target_lon_norm, method='linear')

era_ds = xr.open_dataset(BASE / 'era.nc')
era_lat_full = era_ds['lat'].values
era_lon_full = era_ds['lon'].values

lat_mask = (era_lat_full >= min_lat) & (era_lat_full <= max_lat)
lon_mask = (era_lon_full >= min_lon) & (era_lon_full <= max_lon)

era_da = era_ds[VAR].isel(lat=lat_mask, lon=lon_mask)
era_clim_da = era_ds[CLIM_VAR].isel(lat=lat_mask, lon=lon_mask)
era_times = era_da[TIME_VAR].values
era_data  = era_da.values
era_clim_data  = era_clim_da.values
era_lat   = era_da['lat'].values
era_lon   = era_da['lon'].values

forecast_data = {}
for name in forecasts:
    folder = BASE / name
    if not folder.exists():
        print(f'  skip {name}: no folder')
        continue
    runs = {}
    for f in sorted(folder.glob('*.nc')):
        ds = xr.open_dataset(f)
        da = ds[VAR]
        da = interp_circular_lon(da, era_lat, era_lon)
        runs[f.stem] = {
            'data':  da.values,
            'times': da[TIME_VAR].values,
        }
    forecast_data[name] = runs
    print(f'  {name}: {len(runs)} runs')

era_starts = set(pd.DatetimeIndex(era_times).strftime('%Y%m').unique())
start_sets = [set(r.keys()) for r in forecast_data.values()] + [era_starts]
all_start_dates = sorted(set.intersection(*start_sets))
print(f'common starts: {len(all_start_dates)}')

def find_target(times, start, lead):
    if type_ == 'week':
        lo = start + timedelta(days=7 * lead)
        hi = lo + timedelta(days=7)
    else:
        lo = (start + relativedelta(months=lead)).replace(day=1)
        hi = (lo + relativedelta(months=1))
    lo = np.datetime64(lo)
    hi = np.datetime64(hi)
    mask = (times >= lo) & (times < hi)
    if mask.sum() == 0:
        return None
    return times[mask][0]

def pick(times, data, target):
    idx = np.where(times == np.datetime64(target))[0]
    if len(idx) == 0:
        return None
    return data[idx[0]]

def get_data(entity, image_type, start_str, lead):
    start = datetime.strptime(start_str + '01', '%Y%m%d')

    if entity == 'era':
        target = find_target(era_times, start, lead)
        if target is None:
            return None
        data = pick(era_times, era_data, target)
    else:
        run = forecast_data.get(entity, {}).get(start_str)
        if run is None:
            return None
        target = find_target(run['times'], start, lead)
        if target is None:
            return None
        data = pick(run['times'], run['data'], target)

    if data is None:
        return None

    era_v = pick(era_times, era_data, target)    
    if era_v is not None:
        data = data + era_v*0
    if image_type == 'anomaly':
        return data
    if image_type == 'anomaly_climate':
        clim = pick(era_times, era_clim_data, target)
        if clim is None:
            return None
        return data + clim
    if image_type == 'deviation':
        if entity == 'era':
            return np.zeros_like(data)
        if era_v is None:
            return None
        return data - era_v

def render_map(data, image_type):
    if image_type == 'anomaly_climate':
        vmin, vmax = SCALE_ABS
        cmap = plt.get_cmap('Blues').copy()
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        vmin, vmax = SCALE_ANOM
        cmap = plt.get_cmap('RdBu_r').copy()
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap.set_bad('lightgray')

    fig, ax = plt.subplots(figsize=(4, 2.4), dpi=100)
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto', origin='lower',
                   extent=[era_lon[0], era_lon[-1], era_lat[0], era_lat[-1]])
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='мм')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def render_heatmap(values, row_labels, col_labels, cmap, vmin, vmax, fmt='{:.2f}'):
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=250)
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xlabel('Заблаговременность ' + ('(нед)' if type_ == 'week' else '(мес)'))
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if not np.isnan(v):
                ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=3.5)
    plt.colorbar(im, ax=ax, fraction=0.03)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def compute_metrics(fc, era):
    m = ~np.isnan(fc) & ~np.isnan(era)
    if m.sum() < 10:
        return np.nan, np.nan
    f = fc[m]; o = era[m]
    rmse = float(np.sqrt(((f - o) ** 2).mean()))
    den = np.sqrt((f ** 2).sum() * (o ** 2).sum())
    acc = float((f * o).sum() / den) if den > 0 else np.nan
    return rmse, acc

page1 = {}
raw_metrics = {}
row_labels = all_start_dates
col_labels = [str(l + 1) for l in range(LEADS)]
for name in forecast_data:
    rmse_m = np.full((len(all_start_dates), LEADS), np.nan)
    acc_m  = np.full((len(all_start_dates), LEADS), np.nan)
    for i, sd in enumerate(all_start_dates):
        run = forecast_data[name].get(sd)
        if run is None:
            continue
        start = datetime.strptime(sd + '01', '%Y%m%d')
        for j in range(LEADS):
            target = find_target(era_times, start, j)
            if target is None:
                continue
            fc = pick(run['times'], run['data'], target)
            era_v = pick(era_times, era_data, target)
            if fc is None or era_v is None:
                continue
            rmse, acc = compute_metrics(fc, era_v)
            rmse_m[i, j] = rmse
            acc_m[i, j]  = acc
    raw_metrics[name] = {'rmse': rmse_m, 'acc': acc_m}
    page1[f'{name}|rmse'] = render_heatmap(rmse_m, row_labels, col_labels,
                                           'viridis', *SCALE_RMSE)
    page1[f'{name}|acc']  = render_heatmap(acc_m, row_labels, col_labels,
                                           'RdBu_r', *SCALE_ACC, fmt='{:.3f}')

page2 = {}

def _render_task(args_):
    entity, itype, sd, lead = args_
    data = get_data(entity, itype, sd, lead)
    if data is None or np.all(np.isnan(data)):
        return None
    return (f'{entity}|{itype}|{sd}|{lead}', render_map(data, itype))

all_entities = ['era'] + list(forecast_data.keys()) if args.maps else []
image_types = ['anomaly', 'anomaly_climate', 'deviation']
tasks = [(e, t, s, l)
         for e in all_entities
         for t in image_types
         for s in all_start_dates
         for l in range(LEADS)]

page2 = {}
with ProcessPoolExecutor(max_workers=8) as executor:
    for res in tqdm(executor.map(_render_task, tasks, chunksize=4), total=len(tasks), desc='page2'):
        if res is not None:
            page2[res[0]] = res[1]
    
page3 = {}
names = list(forecast_data.keys())

for a in names:
    for b in names:
        if a == b:
            continue
        for m in ('rmse', 'acc'):
            diff = raw_metrics[a][m] - raw_metrics[b][m]
            key = f'{a}|{b}|{m}'
            page3[key] = render_heatmap(
                diff, row_labels, col_labels,
                'RdBu_r', -SCALE_DIFF[m], SCALE_DIFF[m],
                fmt='{:.2f}' if m == 'rmse' else '{:.3f}'
            )

html = '''<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>SWE comparison</title>
<style>
body{font-family:sans-serif;margin:10px}
.tabs{display:flex;gap:5px;margin-bottom:10px}
.tab{padding:8px 16px;border:1px solid #ccc;background:#f0f0f0;cursor:pointer}
.tab.active{background:#fff;border-bottom:2px solid #000;font-weight:bold}
.page{display:none}.page.active{display:block}
.controls{display:flex;gap:20px;flex-wrap:wrap;margin:10px 0}
.images{display:flex;gap:10px}
.images>div{flex:0 0 50%;max-width:50%}
.images img{width:100%}
.sliders{display:flex;gap:20px;margin-top:10px}
.sliders>div{flex:1}
</style>
</head>
<body>
<div class="tabs">
  <div class="tab active" onclick="tab(1)">Статистика</div>
  <div class="tab" onclick="tab(2)">Сравнение</div>
  <div class="tab" onclick="tab(3)">Статистика разностей</div>
</div>

<div class="page active" id="p1">
  <div class="controls">
    <label>Слева: <select id="s2L"></select></label>
    <label>Справа: <select id="s2R"></select></label>
    <label>Метрика:
      <select id="s2M">
        <option value="rmse">RMSE</option>
        <option value="acc">ACC</option>
      </select>
    </label>
  </div>
  <div class="images">
    <div><img id="i2L"></div>
    <div><img id="i2R"></div>
  </div>
</div>

<div class="page" id="p2">
  <div class="controls">
    <label>Слева: <select id="sL"></select></label>
    <label>Справа: <select id="sR"></select></label>
    <label>Тип:
      <select id="sT">
        <option value="anomaly">Аномалии</option>
        <option value="anomaly_climate">Аномалии + климат</option>
        <option value="deviation">Отклонение от реанализа</option>
      </select>
    </label>
  </div>
  <div class="images">
    <div><img id="iL"></div>
    <div><img id="iR"></div>
  </div>
  <div class="sliders">
    <div>
      <label>Заблаговременность (<span id="leadName">w</span>): <b id="leadVal">1</b></label>
      <input type="range" id="leadSl" style="width:100%">
    </div>
    <div>
      <label>Дата старта: <b id="startVal">-</b></label>
      <input type="range" id="startSl" style="width:100%">
    </div>
  </div>
</div>

<div class="page" id="p3">
  <div class="controls">
    <label>A: <select id="s3A"></select></label>
    <label>B: <select id="s3B"></select></label>
    <label>Метрика:
      <select id="s3M">
        <option value="rmse">RMSE (A − B)</option>
        <option value="acc">ACC (A − B)</option>
      </select>
    </label>
  </div>
  <div class="images">
    <div><img id="i3"></div>
  </div>
</div>

<script>
const P1 = __P1__;
const P2 = __P2__;
const P3 = __P3__;
const ENT = __ENT__;
const SD  = __SD__;
const LEADS = __LEADS__;
const TYPE = "__TYPE__";
const HAS_MAPS = __HAS_MAPS__;

function tab(n){
  document.querySelectorAll('.page').forEach(e=>e.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(e=>e.classList.remove('active'));
  document.getElementById('p'+n).classList.add('active');
  document.querySelectorAll('.tab')[n-1].classList.add('active');
}

const allOpts = ['era'].concat(ENT);
['sL','sR','s2L','s2R'].forEach(id=>{
  const sel=document.getElementById(id);
  allOpts.forEach(o=>{const e=document.createElement('option');e.value=o;e.textContent=o==='era'?'ERA (реанализ)':o;sel.appendChild(e);});
});
['s3A','s3B'].forEach(id=>{
  const sel=document.getElementById(id);
  ENT.forEach(o=>{const e=document.createElement('option');e.value=o;e.textContent=o;sel.appendChild(e);});
});

document.getElementById('sL').value = ENT[0]||'era';
document.getElementById('sR').value = 'era';
document.getElementById('s2L').value = ENT[0]||'era';
document.getElementById('s2R').value = ENT[1]||ENT[0]||'era';
document.getElementById('leadName').textContent = TYPE==='week'?'нед':'мес';
document.getElementById('s3A').value = ENT[0]||'';
document.getElementById('s3B').value = ENT[1]||ENT[0]||'';

const leadSl  = document.getElementById('leadSl');
const startSl = document.getElementById('startSl');
leadSl.max  = LEADS.length; leadSl.min = 1; leadSl.value = 1;
startSl.max = SD.length-1;     startSl.value = 0;

function upd2(){
  const L=document.getElementById('sL').value;
  const R=document.getElementById('sR').value;
  const T=document.getElementById('sT').value;
  const lead = LEADS[+leadSl.value-1];
  const sd   = SD[+startSl.value];
  document.getElementById('leadVal').textContent = lead + 1;
  document.getElementById('startVal').textContent = sd;
  const kL=L+'|'+T+'|'+sd+'|'+lead;
  const kR=R+'|'+T+'|'+sd+'|'+lead;
  document.getElementById('iL').src = P2[kL] ? 'data:image/png;base64,'+P2[kL] : '';
  document.getElementById('iR').src = P2[kR] ? 'data:image/png;base64,'+P2[kR] : '';
}
function upd1(){
  const L=document.getElementById('s2L').value;
  const R=document.getElementById('s2R').value;
  const M=document.getElementById('s2M').value;
  const kL=L+'|'+M, kR=R+'|'+M;
  document.getElementById('i2L').src = P1[kL] ? 'data:image/png;base64,'+P1[kL] : '';
  document.getElementById('i2R').src = P1[kR] ? 'data:image/png;base64,'+P1[kR] : '';
}
function upd3(){
  const A=document.getElementById('s3A').value;
  const B=document.getElementById('s3B').value;
  const M=document.getElementById('s3M').value;
  const k = A+'|'+B+'|'+M;
  document.getElementById('i3').src = P3[k] ? 'data:image/png;base64,'+P3[k] : '';
}


['sL','sR','sT'].forEach(id=>document.getElementById(id).onchange=upd2);
['leadSl','startSl'].forEach(id=>document.getElementById(id).oninput=upd2);
['s2L','s2R','s2M'].forEach(id=>document.getElementById(id).onchange=upd1);
['s3A','s3B','s3M'].forEach(id=>document.getElementById(id).onchange=upd3);

if (!HAS_MAPS) {
  document.querySelectorAll('.tab')[1].style.display = 'none';
}

upd1(); upd2(); upd3();
</script>
</body>
</html>'''

html = (html
        .replace('__P1__', json.dumps(page1, separators=(',', ':')))
        .replace('__P2__', json.dumps(page2, separators=(',', ':')))
        .replace('__P3__', json.dumps(page3, separators=(',', ':'))) 
        .replace('__ENT__', json.dumps(list(forecast_data.keys())))
        .replace('__SD__', json.dumps(all_start_dates))
        .replace('__LEADS__', json.dumps(list(range(LEADS))))
        .replace('__TYPE__', type_)
        .replace('__HAS_MAPS__', 'true' if args.maps else 'false'))

Path(args.out).write_text(html, encoding='utf-8')
print('wrote', args.out, f'({Path(args.out).stat().st_size/1e6:.1f} MB)')

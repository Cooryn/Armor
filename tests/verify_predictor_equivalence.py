"""Compare C++ CSV/TXT exports against the matching Python reference model."""
import argparse
import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

parser=argparse.ArgumentParser()
parser.add_argument('--python-root',type=Path,required=True)
parser.add_argument('--bin-dir',type=Path,required=True)
parser.add_argument('--work-dir',type=Path,required=True)
a=parser.parse_args()
sys.path.insert(0,str(a.python_root))
from tests.reference.predictor.predictor import run_predict
from tests.reference.predictor.predictor_polar import run_predict_polar
from tests.reference.predictor.run_armor import run_predict_armor
root=a.work_dir.resolve();root.mkdir(parents=True,exist_ok=True)
report=[]
def compare(name,path,mode='armor',horizon=50):
    py=root/name/'python';cpp=root/name/'cpp'
    fn={'basic':run_predict,'polar':run_predict_polar,'armor':run_predict_armor}[mode]
    kwargs=dict(prediction_horizon_ms=horizon) if mode=='armor' else {}
    with contextlib.redirect_stdout(io.StringIO()):fn(path,py,**kwargs)
    exe={'basic':'predictor','polar':'predictor_polar','armor':'predictor_armor'}[mode]
    cmd=[str(a.bin_dir.resolve()/(exe+('.exe' if sys.platform == 'win32' else ''))), '--input',str(path),'--output-dir',str(cpp),'--suffix','1']
    if mode=='armor':cmd+=['--prediction-horizon-ms',str(horizon)]
    subprocess.run(cmd,check=True,capture_output=True)
    pf={p.name for p in py.glob('*')};cf={p.name for p in cpp.glob('*')}
    assert pf==cf,(name,pf,cf)
    maximum=0.;count=0
    for file in sorted(pf):
        p,c=py/file,cpp/file
        if p.suffix=='.txt':
            assert p.read_text()==c.read_text(),(name,file,p.read_text(),c.read_text())
            continue
        x,y=pd.read_csv(p),pd.read_csv(c)
        assert list(x.columns)==list(y.columns),(name,file,list(x.columns),list(y.columns))
        assert x.shape==y.shape
        count+=len(x)
        for col in x:
            if pd.api.types.is_numeric_dtype(x[col]) and not pd.api.types.is_bool_dtype(x[col]):
                xx=x[col].to_numpy(dtype=float);yy=y[col].to_numpy(dtype=float)
                np.testing.assert_allclose(xx,yy,rtol=1e-7,atol=1e-8,equal_nan=True,err_msg=f'{name}/{file}/{col}')
                finite=np.isfinite(xx)&np.isfinite(yy)
                if finite.any():maximum=max(maximum,float(np.max(np.abs(xx[finite]-yy[finite]))))
            else:pd.testing.assert_series_equal(x[col],y[col],check_dtype=False)
    report.append(dict(case=name,files=len(pf),csv_rows=count,max_absolute_difference=maximum))
    print(name,'PASS',maximum,flush=True)

for suffix in ('1','2'):
    source=a.python_root/'data'/f'pose_raw_{suffix}.csv'
    for mode in ('basic','polar','armor'):compare(f'real_{suffix}_{mode}',source,mode)
    compare(f'real_{suffix}_zero',source,horizon=0)

# Rotating target: multi-plate observations, duplicates, outliers, missing frames,
# invalid initial rows, ignored quality metadata and wrap-boundary crossings.
rows=[]
for frame in range(90):
    if frame in (7,8,9,42,43):continue
    for aid in ([0,1,1] if frame%3==0 else [frame%4]):
        theta=frame*.12+aid*np.pi/2
        x=.1+.26*np.sin(theta);y=.2+(aid%2)*.015;z=3-.26*np.cos(theta)
        row=dict(frame_id=frame,timestamp=frame*1000/30,x=x,y=y,z=z,target_yaw=np.arctan2(x,z),target_pitch=np.arctan2(y,np.hypot(x,z)),distance=np.linalg.norm([x,y,z]),armor_orientation_yaw=(theta+np.pi)%(2*np.pi)-np.pi,detection_score=.4+.1*(frame%6),reprojection_error=(frame%4)*.5)
        if frame==0:row['distance']=-1
        if frame==26:row['distance']+=3
        if frame==50:row['target_yaw']=np.nan
        rows.append(row)
synthetic=root/'synthetic.csv';pd.DataFrame(rows).to_csv(synthetic,index=False)
compare('synthetic',synthetic)
armor_rows = pd.read_csv(root/'synthetic/cpp/armor_prediction_result_1.csv')
unmatched = armor_rows[armor_rows.status == 'prediction_only']
assert not unmatched.empty and unmatched.armor_id.eq(-1).all()
assert unmatched[['xa', 'za', 'pred_armor_yaw', 'obs_armor_yaw',
                  'err_target_yaw', 'err_target_pitch', 'err_distance', 'err_armor_yaw']].isna().all().all()
assert np.isfinite(unmatched[['xc', 'yc', 'zc', 'future_xc', 'future_yc', 'future_zc']]).all().all()
armor_future = pd.read_csv(root/'synthetic/cpp/armor_future_prediction_1.csv')
assert (armor_future.groupby('frame_id').armor_id.nunique() == 4).all()
assert np.isfinite(armor_future[['x', 'y', 'z', 'armor_orientation_yaw']]).all().all()
compare('synthetic_single_plate',synthetic,'basic')
for size in (0,1):
    path=root/f'small_{size}.csv';pd.DataFrame(rows).iloc[:size].to_csv(path,index=False)
    for mode in ('basic','polar','armor'):compare(f'small_{size}_{mode}',path,mode)
# Timestamp and frame validation must reject exactly the same fixtures.
valid=pd.DataFrame(rows).iloc[3:8].copy()
for case in ('timestamp_decreases','inconsistent_frame_time','negative_frame','fractional_frame','nan_time'):
    d=valid.copy()
    if case=='timestamp_decreases':d.loc[d.index[-1],'timestamp']=-10
    if case=='inconsistent_frame_time':d.loc[d.index[-1],'timestamp']+=1
    if case=='negative_frame':d.loc[d.index[-1],'frame_id']=-1
    if case=='fractional_frame':d['frame_id']=d.frame_id.astype(float);d.loc[d.index[-1],'frame_id']=1.5
    if case=='nan_time':d.loc[d.index[-1],'timestamp']=np.nan
    path=root/f'{case}.csv';d.to_csv(path,index=False)
    try:run_predict_armor(path,root/'invalid_python')
    except ValueError:pass
    else:raise AssertionError(case+' Python accepted')
    r=subprocess.run([str(a.bin_dir.resolve()/('predictor_armor.exe' if sys.platform == 'win32' else 'predictor_armor')),'--input',str(path),'--output-dir',str(root/'invalid_cpp')],capture_output=True)
    assert r.returncode!=0,case
    report.append(dict(case=case,rejection='matched'))
def basic_row(frame, timestamp, position):
    x, y, z = position
    return dict(frame_id=frame, timestamp=timestamp, x=x, y=y, z=z,
                target_yaw=np.arctan2(x, z), target_pitch=np.arctan2(y, np.hypot(x, z)),
                distance=np.linalg.norm(position), note='quoted, "value"\ncontinuation')

# Out-of-order frames, invalid nearer rows, equal distances, missing observations,
# repeated/decreasing positive timestamps, BOM, CRLF and quoted multiline extra fields.
single_rows = [
    basic_row(6, 500, (.1, .1, 3.2)),
    basic_row(0, 0, (0, 0, 4)), basic_row(0, 0, (0, 0, 3)),
    dict(basic_row(1, 100, (0, 0, .1)), distance=-1),
    dict(basic_row(1, 100, (0, 0, .1)), target_yaw=np.nan),
    dict(basic_row(1, 100, (0, 0, .1)), x=np.nan),
    basic_row(1, 100, (1, 0, 3)), basic_row(1, 100, (-1, 0, 3)),
    dict(basic_row(2, 200, (0, 0, 3)), target_pitch=np.nan),
    basic_row(3, 300, (.7, .1, 3.3)), basic_row(4, 300, (.8, .1, 3.4)),
    basic_row(5, 250, (.9, .1, 3.5)),
]
single = pd.DataFrame(single_rows).astype({'frame_id': float, 'timestamp': float})
single_path = root/'single_plate_selection.csv'
single.to_csv(single_path, index=False, encoding='utf-8-sig', lineterminator='\r\n')
compare('single_plate_selection', single_path, 'basic')
selection = pd.read_csv(root/'single_plate_selection/cpp/prediction_result_1.csv')
assert selection.frame_id.tolist() == [1, 3, 4, 5, 6]
assert selection.observed_x.iloc[0] == 1 and selection.predicted_z.iloc[0] == 3
single_frame = root/'single_valid_frame.csv'
pd.DataFrame([basic_row(0, 0, (0, 0, 3))]).to_csv(single_frame, index=False)
compare('single_valid_frame', single_frame, 'basic')

base = pd.DataFrame([basic_row(i, i*40, (.03-i*.002, .2, -3)) for i in range(40)])
base['coordinate_frame'] = 'base'
origin_names = ['base_reference_timestamp_ms', 'base_origin_x_m', 'base_origin_y_m', 'base_origin_z_m']
for name, value in zip(origin_names, [0, .1, .2, .3]):
    base[name] = value
base_path = root/'single_plate_base.csv'
base.to_csv(base_path, index=False)
compare('single_plate_base', base_path, 'basic')
base_result = pd.read_csv(root/'single_plate_base/cpp/prediction_result_1.csv')
assert base_result.coordinate_frame.eq('base').all()
assert base_result.error_yaw.abs().max() < .01
assert base_result.columns.tolist()[-5:] == ['coordinate_frame', *origin_names]

armor_base = base.copy()
armor_base['armor_orientation_yaw'] = .3
armor_base_path = root/'armor_base.csv'
armor_base.to_csv(armor_base_path, index=False)
compare('armor_base', armor_base_path)
base_future = pd.read_csv(root/'armor_base/cpp/armor_future_prediction_1.csv')
assert base_future.coordinate_frame.eq('base').all()
np.testing.assert_allclose(np.linalg.norm(base_future[['qw', 'qx', 'qy', 'qz']], axis=1), 1, atol=1e-12)

basic_exe = str(a.bin_dir.resolve()/('predictor.exe' if sys.platform == 'win32' else 'predictor'))
for case in ('negative_frame', 'fractional_frame', 'nan_frame', 'negative_time', 'nan_time',
             'missing_pitch', 'mixed_frame', 'unknown_frame', 'missing_origin',
             'mixed_origin', 'nan_origin', 'negative_reference'):
    d = base.copy()
    d['frame_id'] = d.frame_id.astype(float)
    if case == 'negative_frame': d.loc[0, 'frame_id'] = -1
    if case == 'fractional_frame': d.loc[0, 'frame_id'] = .5
    if case == 'nan_frame': d.loc[0, 'frame_id'] = np.nan
    if case == 'negative_time': d.loc[0, 'timestamp'] = -1
    if case == 'nan_time': d.loc[0, 'timestamp'] = np.nan
    if case == 'missing_pitch': d = d.drop(columns='target_pitch')
    if case == 'mixed_frame': d.loc[1, 'coordinate_frame'] = 'camera'
    if case == 'unknown_frame': d.loc[0, 'coordinate_frame'] = 'world'
    if case == 'missing_origin': d = d.drop(columns='base_origin_x_m')
    if case == 'mixed_origin': d.loc[1, 'base_origin_z_m'] += 1
    if case == 'nan_origin': d.loc[0, 'base_origin_y_m'] = np.nan
    if case == 'negative_reference': d['base_reference_timestamp_ms'] = -1
    path = root/f'single_plate_invalid_{case}.csv'
    d.to_csv(path, index=False)
    try:
        run_predict(path, root/f'invalid_basic_python_{case}')
    except ValueError:
        pass
    else:
        raise AssertionError(case+' Python accepted')
    rejection = subprocess.run([basic_exe, '--input', str(path), '--output-dir',
                                str(root/f'invalid_basic_cpp_{case}')], capture_output=True)
    assert rejection.returncode == 2, (case, rejection.stderr)
    report.append(dict(case='single_plate_'+case, rejection='matched'))

for arguments in (['--fixed-noise'], ['--prediction-horizon-ms', '50']):
    rejection = subprocess.run([basic_exe, *arguments], capture_output=True)
    assert rejection.returncode == 2 and b'Unknown argument' in rejection.stderr
    report.append(dict(case='single_plate_removed_'+arguments[0], rejection='matched'))

from plot.basic import plot_basic
cpp_results = root/'real_2_basic/cpp'
snapshot = {p.name: p.read_bytes() for p in cpp_results.iterdir()}
with contextlib.redirect_stdout(io.StringIO()):
    plot_basic(cpp_results/'prediction_result_1.csv', a.python_root/'data/pose_raw_2.csv',
               root/'basic_plots', suffix='1')
assert {p.name for p in (root/'basic_plots').iterdir()} == {
    'prediction_error_curve_1.png', 'top_down_trajectory_1.png'}
assert all(p.stat().st_size > 1000 for p in (root/'basic_plots').iterdir())
assert snapshot == {p.name: p.read_bytes() for p in cpp_results.iterdir()}
report.append(dict(case='single_plate_plot_compatibility', result='passed'))
(root/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('All',len(report),'cases passed')

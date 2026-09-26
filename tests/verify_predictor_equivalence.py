"""Compare C++ CSV/TXT exports against the original Python implementation."""
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
def compare(name,path,mode='armor',fixed=False,horizon=50):
    py=root/name/'python';cpp=root/name/'cpp'
    fn={'basic':run_predict,'polar':run_predict_polar,'armor':run_predict_armor}[mode]
    kwargs=dict(adaptive_noise=not fixed,prediction_horizon_ms=horizon) if mode=='armor' else {}
    with contextlib.redirect_stdout(io.StringIO()):fn(path,py,**kwargs)
    exe={'basic':'predictor','polar':'predictor_polar','armor':'predictor_armor'}[mode]
    cmd=[str(a.bin_dir.resolve()/(exe+('.exe' if sys.platform == 'win32' else ''))), '--input',str(path),'--output-dir',str(cpp),'--suffix','1','--prediction-horizon-ms',str(horizon)]
    if fixed:cmd+=['--fixed-noise']
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
    compare(f'real_{suffix}_fixed',source,fixed=True)
    compare(f'real_{suffix}_zero',source,horizon=0)

# Rotating target: multi-plate observations, duplicates, outliers, missing frames,
# invalid initial rows, adaptive metadata and wrap-boundary crossings.
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
for fixed in (False,True):compare(f'synthetic_{fixed}',synthetic,fixed=fixed)
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
(root/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('All',len(report),'cases passed')

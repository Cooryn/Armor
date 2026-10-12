"""Render old/new single-plate filters on isolated video fragments and verify native replay."""
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tests.tune_single_noise import metrics, replay, video_segments
from tests.evaluate_single_response import evaluate, trajectory


def main():
    work = Path('tests/outputs/single-noise')
    report = dict(parameters=dict(q=100, angle_sd=.005, range_sd=.1), videos=[],
                  simulation=dict(baseline=evaluate(.01, .04, .4), tuned=evaluate(100, .005, .1)))
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    for i in (1, 2):
        data = pd.read_csv(work / f'observations_{i}.csv')
        native = data
        full = replay(data, 100, .005, .1)
        np.testing.assert_allclose(full, native[['x', 'vx', 'y', 'vy', 'z', 'vz']], atol=1e-9, rtol=1e-9)
        old, new = replay(data, .01, .04, .4, True), replay(data, 100, .005, .1, True)
        report['videos'].append(dict(video=i, frames=len(data), observations=int(data.yaw.notna().sum()),
                                     baseline=metrics(data, old, 0, len(data)), tuned=metrics(data, new, 0, len(data))))
        mask = (data.timestamp >= (1000 if i == 1 else 10000)) & (data.timestamp < (2000 if i == 1 else 11500))
        ax = axes[i-1]
        ax.plot(data.timestamp[mask] / 1000, data.yaw[mask], 'k.', label='Observation')
        for states, label in ((old, 'Old'), (new, 'Tuned')):
            ax.plot(data.timestamp[mask] / 1000, np.arctan2(states[mask, 0], states[mask, 4]), label=label)
        ax.set(title=f'Video {i}: isolated fragments (both filters restart at fragment boundaries)', ylabel='Yaw / rad')
        ax.legend(loc='upper right')
        cap = cv2.VideoCapture(f'assets/video/video_{i}.avi')
        width, height = int(cap.get(3)), int(cap.get(4))
        size = (640, round(height * 640 / width))
        size = (size[0], size[1] // 2 * 2)
        writer = cv2.VideoWriter(str(work / f'comparison_{i}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                 cap.get(5), (size[0]*2, size[1]))
        assert writer.isOpened()
        if i == 1:
            matrix = np.array([[1286.307063384126, 0, 645.34450819155256], [0, 1288.1400736562441, 483.6163720308021], [0, 0, 1.]])
            distortion = np.array([-.47562935060124745, .21831745829617311, .0004957613589406044, -.00034617769548693592, 0])
        else:
            matrix = np.array([[1711.311186, 0, 732.488057], [0, 1714.616882, 546.930868], [0, 0, 1.]])
            distortion = np.array([-.119922, -.078593, .007511, -.028028, 0])
        segments = video_segments(data)
        for frame, row in enumerate(data.itertuples()):
            ok, original = cap.read()
            assert ok
            panels = []
            observed = row.distance * np.array([np.cos(row.pitch)*np.sin(row.yaw), np.sin(row.pitch), np.cos(row.pitch)*np.cos(row.yaw)])
            for states, label in ((old, 'OLD'), (new, 'TUNED')):
                canvas = original.copy()
                points = [observed, states[frame, ::2], states[frame, ::2] + .05 * states[frame, 1::2]]
                for point, color, radius in zip(points, [(255,255,255), (50,255,50), (0,190,255)], [10, 7, 4]):
                    if np.isfinite(point).all() and point[2] > .1:
                        pixel = cv2.projectPoints(point[None], np.zeros(3), np.zeros(3), matrix, distortion)[0].ravel()
                        if np.max(np.abs(pixel)) < 10000:
                            cv2.circle(canvas, tuple(np.rint(pixel).astype(int)), radius, color, 2)
                canvas = cv2.resize(canvas, size)
                cv2.rectangle(canvas, (0,0), (size[0],82), (0,0,0), -1)
                for line, text in enumerate([f'{label} | frame {frame} | fragment {segments[frame]}',
                                             'White: observed  Green: current  Orange: +50ms',
                                             'Fragment replay; restart BOTH filters at boundaries']):
                    cv2.putText(canvas, text, (10, 23 + line*24), cv2.FONT_HERSHEY_SIMPLEX, .47, (255,255,255), 1, cv2.LINE_AA)
                panels.append(canvas)
            combined = np.hstack(panels)
            writer.write(combined)
            if frame == (35 if i == 1 else 305):
                cv2.imwrite(str(work / f'comparison_{i}.png'), combined)
        cap.release()
        writer.release()
        check = cv2.VideoCapture(str(work / f'comparison_{i}.mp4'))
        assert int(check.get(cv2.CAP_PROP_FRAME_COUNT)) == len(data)
        check.release()
    t = np.arange(720) / 60
    truth = trajectory(t, True)
    obs = np.column_stack([np.arctan2(truth[:,0], truth[:,2]), np.arctan2(truth[:,1], np.hypot(truth[:,0],truth[:,2])), np.linalg.norm(truth,axis=1)])
    obs += np.random.default_rng(20261071).normal(size=obs.shape) * [.003,.001,.035]
    data = pd.DataFrame(obs, columns=['yaw','pitch','distance'])
    data['timestamp'] = t*1000
    axes[2].plot(t, truth[:,0], 'k', label='Truth')
    for parameters, label in (((.01,.04,.4),'Old'), ((100,.005,.1),'Tuned')):
        axes[2].plot(t, replay(data,*parameters)[:,0], label=label)
    axes[2].set(title='60 FPS single-plate simulation (not live camera measurement)', xlim=(2,6), xlabel='Time / s', ylabel='X / m')
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(work / 'comparison.png', dpi=150)
    (work / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

"""Summarize parameter diagnostics; counts are not annotated precision/recall."""
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('tests/outputs/detector-tuning'))
    parser.add_argument('--candidate', type=int, default=25)
    parser.add_argument('--render-video', action='store_true', help='Requires full-frame evaluation (stride 1)')
    args = parser.parse_args()
    data = pd.read_csv(args.directory / 'detections.csv')
    frames = data.drop_duplicates(['source', 'frame', 'variant', 'candidate']).copy()
    frames['has_detection'] = frames['count'] > 0
    summary = frames.groupby(['source', 'variant', 'candidate']).agg(frames=('count', 'size'),
        detected_frames=('has_detection', 'sum'), detections=('count', 'sum')).reset_index()
    summary.to_csv(args.directory / 'summary.csv', index=False)
    print(summary[summary.source.str.startswith('video')].to_string(index=False))
    photos = summary[summary.source.str.startswith('image')]
    print(photos.groupby(['variant','candidate'])[['detected_frames','detections']].sum().to_string())
    for variant in ('original', 'dim', 'small', 'wrong_color'):
        rows = []
        for i in range(11):
            name = f'image_{i}'
            original = cv2.imread(f'assets/image/{name}.jpg')
            if variant == 'dim': original = cv2.convertScaleAbs(original, alpha=.6)
            if variant == 'small': original = cv2.resize(original, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
            panels = []
            for candidate in (0, args.candidate):
                canvas = cv2.resize(original, (480, 320))
                selected = data[(data.source == name) & (data.variant == variant) & (data.candidate == candidate) & (data['index'] >= 0)]
                for row in selected.itertuples():
                    corners = np.array([[getattr(row, f'x{k}')*480, getattr(row, f'y{k}')*320] for k in range(4)]).astype(np.int32)
                    cv2.polylines(canvas, [corners], True, (0,255,0), 2)
                    cv2.putText(canvas, f'{row.score:.2f}', tuple(corners[0]), 0, .4, (0,255,255), 1)
                cv2.rectangle(canvas, (0,0), (480,25), (0,0,0), -1)
                cv2.putText(canvas, f'{name} {variant} | candidate {candidate} | count {len(selected)}', (5,18), 0, .45, (255,255,255), 1)
                panels.append(canvas)
            rows.append(np.hstack(panels))
        for start in (0, 6):
            cv2.imwrite(str(args.directory / f'{variant}_{start}_candidate{args.candidate}.jpg'), np.vstack(rows[start:start+6]))
    if args.render_video:
        for i in (1, 2):
            source = f'video_{i}'
            groups = {(int(frame), int(candidate)): group for (frame, candidate), group in
                      data[data.source == source].groupby(['frame', 'candidate'])}
            cap = cv2.VideoCapture(f'assets/video/{source}.avi')
            assert cap.isOpened()
            width = 640
            height = int(cap.get(4) * width / cap.get(3)) // 2 * 2
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            destination = args.directory / f'{source}_comparison.mp4'
            writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*'mp4v'), cap.get(5), (width*2, height))
            assert writer.isOpened()
            for frame in range(total):
                ok, original = cap.read()
                assert ok
                panels = []
                for candidate, label in ((0, 'Before'), (args.candidate, 'After')):
                    canvas = cv2.resize(original, (width, height))
                    selected = groups[(frame, candidate)]
                    for row in selected[selected['index'] >= 0].itertuples():
                        corners = np.array([[getattr(row,f'x{k}')*width, getattr(row,f'y{k}')*height] for k in range(4)], np.int32)
                        cv2.polylines(canvas, [corners], True, (0,255,0), 2)
                    cv2.rectangle(canvas, (0,0), (width,32), (0,0,0), -1)
                    cv2.putText(canvas, f'{label} | frame {frame} | detections {int(selected.iloc[0]["count"])}', (10,23), 0, .6, (255,255,255), 1)
                    panels.append(canvas)
                writer.write(np.hstack(panels))
            writer.release()
            cap.release()
            check = cv2.VideoCapture(str(destination))
            assert int(check.get(cv2.CAP_PROP_FRAME_COUNT)) == total
            check.release()


if __name__ == '__main__':
    main()

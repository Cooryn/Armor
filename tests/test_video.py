from collections import deque
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest

import cv2
import numpy as np
import pandas as pd

from plot.video import PROFILES, project, draw_frame, render_video, compute_plate_corners


class VideoTests(unittest.TestCase):
    def test_base_projection_transforms_full_plate_corners(self):
        rotation, _ = cv2.Rodrigues(np.array([.15, -.2, .1]))
        translation = np.array([.1, -.05, .2])
        inverse = np.column_stack((rotation.T, -rotation.T @ translation))
        state = dict(status="updated", w=0., xc=.1, yc=.2, zc=3., body_yaw=.3, r=.26, dl=.03, dh=.02,
                     prediction_horizon_ms=50, prediction_timestamp=50,
                     future_xc=.15, future_yc=.2, future_zc=3.05, future_body_yaw=.4)
        outlines = []
        def record(canvas, corners, *args, **kwargs):
            outlines.append(corners.copy())
            return []
        matrix, distortion, size = PROFILES['2']
        with patch('plot.video.draw_plate_outline', side_effect=record):
            draw_frame(np.zeros((size[1], size[0], 3), np.uint8), 0, 30, state, [], {},
                       matrix, distortion, deque(maxlen=30), inverse)
        self.assertEqual(len(outlines), 8)
        for future in (False, True):
            values = [.15, .2, 3.05, .4, .26, .03, .02] if future else [.1, .2, 3., .3, .26, .03, .02]
            for aid in range(4):
                base = compute_plate_corners(*values, aid)
                expected = (rotation.T @ (base - translation).T).T
                np.testing.assert_allclose(outlines[aid + (4 if future else 0)], expected, atol=1e-12)

    def test_second_camera_projection(self):
        matrix, distortion, size = PROFILES['2']
        expected, _ = cv2.projectPoints(np.array([[.1,.2,3.]]), np.zeros(3), np.zeros(3), matrix, distortion)
        self.assertEqual(project(.1,.2,3,*size,matrix,distortion), tuple(expected.ravel().astype(int)))
        self.assertIsNone(project(0,0,-3,*size,matrix,distortion))

    def test_no_state_clears_trail(self):
        matrix, distortion, size = PROFILES['2']
        trail = deque([(20,20)],maxlen=30)
        source = np.zeros((size[1],size[0],3), np.uint8)
        canvas = draw_frame(source,2,30,None,[],{},matrix,distortion,trail,np.eye(4)[:3])
        self.assertEqual(len(trail),0)
        self.assertEqual(canvas.shape, source.shape)
        np.testing.assert_array_equal(canvas[120:], source[120:])

    def test_frame_alignment_and_no_forward_fill(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video, output = root/'source.avi',root/'output.mp4'
            writer = cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),24,(1280,1024))
            self.assertTrue(writer.isOpened())
            for i in range(4):
                writer.write(np.full((1024,1280,3),i*30,np.uint8))
            writer.release()
            pred = root/'pred.csv'
            raw = root/'raw.csv'
            pd.DataFrame([dict(frame_id=i,xc=0,yc=0,zc=3,body_yaw=0,r=.26,dl=0,dh=0,w=0,status='updated',
                               prediction_horizon_ms=0,prediction_timestamp=i*1000/24,
                               future_xc=0,future_yc=0,future_zc=3,future_body_yaw=0,coordinate_frame='base')
                          for i in [1,3]]).to_csv(pred,index=False)
            pd.DataFrame(columns=['frame_id','x','y','z']).to_csv(raw,index=False)
            poses = root/'poses.csv'
            pd.DataFrame([dict(frame_id=i, qw=1, qx=0, qy=0, qz=0, x=0, y=0, z=0)
                          for i in range(4)]).to_csv(poses,index=False)
            seen = []
            def recording(frame,fid,fps,state,*args):
                seen.append((fid,None if state is None else state['frame_id']))
                return draw_frame(frame,fid,fps,state,*args)
            with patch('plot.video.draw_frame',side_effect=recording):
                self.assertEqual(render_video(video,pred,raw,output,poses,profile='2'),4)
            self.assertEqual(seen,[(0,None),(1,1),(2,None),(3,3)])
            cap = cv2.VideoCapture(str(output))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),4)
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),1280)
            self.assertEqual(cap.get(cv2.CAP_PROP_FPS),24)
            cap.release()
            with self.assertRaises(ValueError):
                render_video(video,pred,raw,video,poses,profile='2')


if __name__ == '__main__':
    unittest.main()

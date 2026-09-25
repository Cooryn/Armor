from collections import deque
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest

import cv2
import numpy as np
import pandas as pd

from plot.video import PROFILES, project, draw_frame, render_video


class VideoTests(unittest.TestCase):
    def test_second_camera_projection(self):
        matrix, distortion, size = PROFILES['2']
        expected, _ = cv2.projectPoints(np.array([[.1,.2,3.]]), np.zeros(3), np.zeros(3), matrix, distortion)
        self.assertEqual(project(.1,.2,3,*size,matrix,distortion), tuple(expected.ravel().astype(int)))
        self.assertIsNone(project(np.nan,0,3))

    def test_no_state_clears_trail(self):
        matrix, distortion, size = PROFILES['2']
        trail = deque([(20,20)],maxlen=30)
        source = np.zeros((size[1],size[0],3), np.uint8)
        canvas = draw_frame(source,2,30,None,[],{},matrix,distortion,trail)
        self.assertEqual(len(trail),0)
        np.testing.assert_array_equal(canvas[:,:size[0]],source)

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
            pd.DataFrame([dict(frame_id=i,xc=0,yc=0,zc=3,body_yaw=0,r=.26,dl=0,dh=0,status='updated')
                          for i in [1,3]]).to_csv(pred,index=False)
            pd.DataFrame(columns=['frame_id','x','y','z']).to_csv(raw,index=False)
            seen = []
            def recording(frame,fid,fps,state,*args):
                seen.append((fid,None if state is None else state['frame_id']))
                return draw_frame(frame,fid,fps,state,*args)
            with patch('plot.video.draw_frame',side_effect=recording):
                self.assertEqual(render_video(video,pred,raw,output,profile='2'),4)
            self.assertEqual(seen,[(0,None),(1,1),(2,None),(3,3)])
            cap = cv2.VideoCapture(str(output))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),4)
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),1620)
            self.assertEqual(cap.get(cv2.CAP_PROP_FPS),24)
            cap.release()
            with self.assertRaises(ValueError):
                render_video(video,pred,raw,video,profile='2')
            with self.assertRaises(ValueError):
                render_video(video,pred,raw,output,profile='1')


if __name__ == '__main__':
    unittest.main()

import cv2
import numpy as np
import random
import math

import mock
import unittest

from dot_to_dot_generator import *


class TestProgram(unittest.TestCase):
    def test_drawline(self):
        canvas = np.array([[0,0,0],[0,0,0],[0,0,0]])
        draw_line(canvas, [0,0],[2,2], 100)
        canvas = canvas.astype(np.int16, copy=False)
        self.assertTrue(str(canvas) == str(np.array([[141,0,0],[0,141,0],[0,0,0]])))
        draw_line(canvas, [0,0],[0,2], 100)
        self.assertTrue(str(canvas) == str(np.array([[241,100,0],[0,141,0],[0,0,0]])))
    def test_coremeta(self):
        class A():
            __metaclass__=CoreMetaGen('f')
            A = 1
        class B(A):
            B = 2
        class C(B):
            C = 4
        class D(A):
            D = 2
        self.assertEquals(set(A._subclasses.keys()), set(['B', 'C', 'D']))
        self.assertEquals(set(A._subclasses.itervalues()), set([B, C, D]))
    def test_actions(self):
        actions = [Left, Right, Up, Down, Forward, Backward, Supress, Extremize]
        points = [[1,1],[2,2],[2,1],[3,4],[4,2],[2,1]]
        kwargs = {"step":2, "factor":0.5}
        results = {0:[[1,-1],[1,3],[-1,1],[3,1],[1.5,1.5],     None,      None,     None],
                   1:[[2, 0],[2,4],[ 0,2],[4,2],[  2,1.5],[1.5,1.5],[1.75,1.5],[2.25,2.5]],
                   2:[[2,-1],[2,3],[ 0,1],[4,1],[2.5,2.5],[  2,1.5],[2.25,2.0],[1.75,0.0]],
                   5:[[2,-1],[2,3],[ 0,1],[4,1],     None,[  3,1.5],      None,     None]}
        for i,a in enumerate(actions):
            z = a(points=points, **kwargs)
            for k in results.keys():
                if results[k][i] == None:
                    self.assertRaises(IgnorableException, lambda:z.get(k))
                else:
                    self.assertEqual(z.get(k), results[k][i])
    def test_filters(self):
        filters = [BoundaryFilter, RepeatFilter, DescretiseFilter]
        params = {"border":2, "width":30, "height":30, "step":2}
        points = [[1,1],[2,2],[2,1],[3,4],[4,2],[2,1]]
        trial_points = [[1,2],[-5,-5],[3,4],[2,3],[100,3]]
        outputs = [[None,None,[3,4],[2,3],None],
                   [[1,2],[-5,-5],None,[2,3],[100,3]],
                   [[0,2],[-6,-6],[2,4],[2,2],[100,2]]]
        for i,a in enumerate(filters):
            z = a(points=points, **params)
            for k in range(len(trial_points)):
                self.assertEqual(z.doFilter(trial_points[k]), outputs[i][k])
    def test_kernels(self):
        g = GaussianKernel(4,3,1.0)
        gg = g.gen() * 100
        gg = gg.astype(np.int8, copy=False)
        self.assertEqual(str(gg.tolist()), str([[ 1, 3, 1],[ 9,25, 9],[ 9,25, 9],[ 1, 3, 1]]))
        l = LorentzianKernel(4,3,1.0)
        ll = l.gen() * 100
        ll = ll.astype(np.int8, copy=False)
        self.assertEqual(str(ll.tolist()), str([[ 4, 6, 4],[ 9,16, 9],[ 9,16, 9],[ 4, 6, 4]]))
    @mock.patch('dot_to_dot_generator.loadJSON')
    def test_point_loaders(self, m):
        m.return_value = [[4,4],[4,4],[4,4],[4,4],[4,4]]
        loaders = [RandomPointLoader, TotalRandomPointLoader, JSONPointLoader]
        params = [{"length":5, "width":10, "height":15, "border":2}, {"length":5, "width":10, "height":10, "border":2}, {"filename":"baz.bar"}]
        for i,l in enumerate(loaders):
            z = l(**params[i])
            a = z.get()
            self.assertTrue(len(a) == 5)
            for a in z.get():
                self.assertTrue(a[1]<8)
                self.assertTrue(a[1]>1)
                self.assertTrue(a[0]<13)
                self.assertTrue(a[0]>1)
    def test_metrics(self):
        img = 255 - 200*np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
        b = LinearBlurCompare(img, 'GaussianKernel', 3, 3, 1)
        self.assertEqual(int(1000 * b.kernel[1,1] / b.kernel[0,1]), 2718)
        self.assertEqual(str((100*b.kernel).astype(np.int32, copy=True).tolist()), str([[4, 12, 4], [12, 33, 12], [4, 12, 4]]))
        self.assertEqual(str((b.img*100).astype(np.int32, copy=True).tolist()), str([[0, 0, 0, 0, 0, 0],[0, 44, 121, 44, 0, 0],[0, 121, 374, 242, 44, 0],[0, 44, 242, 374, 121, 0],[0, 0, 44, 121, 44, 0],[0, 0, 0, 0, 0, 0]]))
        self.assertEqual(np.sum(b.canvas), 0)
        self.assertEqual(int(100000*b.get([[2,2],[4,4]])), 0)
        
        b = RMSBlurCompare(img, 'GaussianKernel', 3, 3, 1)
        self.assertEqual(int(1000 * b.kernel[1,1] / b.kernel[0,1]), 2718)
        self.assertEqual(str((100*b.kernel).astype(np.int32, copy=True).tolist()), str([[4, 12, 4], [12, 33, 12], [4, 12, 4]]))
        self.assertEqual(str((b.img*100).astype(np.int32, copy=True).tolist()), str([[0, 0, 0, 0, 0, 0],[0, 44, 121, 44, 0, 0],[0, 121, 374, 242, 44, 0],[0, 44, 242, 374, 121, 0],[0, 0, 44, 121, 44, 0],[0, 0, 0, 0, 0, 0]]))
        self.assertEqual(np.sum(b.canvas), 0)
        self.assertEqual(int(100000*b.get([[2,2],[4,4]])), 0)
        
        b = RMSBlurCompare(img, 'LorentzianKernel', 3, 3, 1)
        self.assertEqual(int(1000 * b.kernel[1,1] / b.kernel[0,1]), 2718)
        self.assertEqual(str((100*b.kernel).astype(np.int32, copy=True).tolist()), str([[7, 10, 7], [10, 29, 10], [7, 10, 7]]))
        self.assertEqual(str((b.img*100).astype(np.int32, copy=True).tolist()), str([[0, 0, 0, 0, 0, 0], [0, 75, 113, 75, 0, 0], [0, 113, 385, 227, 75, 0], [0, 75, 227, 385, 113, 0], [0, 0, 75, 113, 75, 0], [0, 0, 0, 0, 0, 0]]))
        self.assertEqual(np.sum(b.canvas), 0)
        self.assertEqual(int(100000*b.get([[2,2],[4,4]])), 0)
        
        a = AvgLengthCompare(None, 5)
        self.assertEqual(a.get([[0,0],[0,5],[5,5],[10,5]]), 0)
        self.assertEqual(a.get([[0,0],[0,5],[5,5],[10,10]]), (5.0*math.sqrt(2)-5.0)/3.0)
        self.assertEqual(a.get([[0,0],[0,5],[10,5],[10,5]]), 10.0/3.0)
        self.assertEqual(a.get([[0,0],[0,0],[0,0],[0,0]]), 5.0)
        self.assertEqual(a.get([[10,0],[10,10],[0,10],[0,0]]), 5.0)
    @mock.patch('cv2.imwrite')
    def test_outputters(self, m):
        o = ImageOutputter(12,12,'cat-{}.png')
        o.output([[4,4],[7,7],[10,0]], 5)
        self.assertEqual(len(m.mock_calls), 1)
        self.assertEqual(m.mock_calls[0][1][0], "cat-5.png")
        self.assertEqual(str(m.mock_calls[0][1][1].astype(np.int32, copy=True).tolist()), str(
        [[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
         [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
         [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
         [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
         [255, 255, 255, 255,   0, 255, 255, 255, 255, 255, 255, 255],
         [255, 255, 255, 255, 255,   0, 255, 255, 255, 255, 255, 255],
         [255, 255, 255, 255, 255, 255,   0, 255, 255, 255, 255, 255],
         [255, 255, 255, 255, 255,  58,  58,  58, 255, 255, 255, 255],
         [255, 255, 255,  58,  58, 255, 255, 255, 255, 255, 255, 255],
         [255,  58,  58, 255, 255, 255, 255, 255, 255, 255, 255, 255],
         [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
         [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]]
        ))
        o = BlurredImageOutputter(12,12,'catd-{}.png', kernel_name="LorentzianKernel",k_width=5,k_height=5,r=1.2)
        o.kernel = np.array([[0.25,0.25],[0.25,0.25]], np.float32)
        o.output([[4,4],[7,7],[10,0]], 5)
        self.assertEqual(len(m.mock_calls), 2)
        self.assertEqual(m.mock_calls[1][1][0], "catd-5.png")
        self.assertEqual(str(m.mock_calls[1][1][1].astype(np.int32, copy=True).tolist()), str(
        [[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
         [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
         [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
         [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 
         [255, 255, 255, 255, 154, 154, 255, 255, 255, 255, 255, 255], 
         [255, 255, 255, 255, 154,  54, 154, 255, 255, 255, 255, 255], 
         [255, 255, 255, 255, 255, 154,  54, 154, 255, 255, 255, 255], 
         [255, 255, 255, 255, 255, 177,   0,   0, 177, 255, 255, 255], 
         [255, 255, 255, 177, 100, 100, 100, 100, 177, 255, 255, 255], 
         [177, 177, 100, 100, 100, 177, 255, 255, 255, 255, 255, 255], 
         [177, 177, 100, 177, 255, 255, 255, 255, 255, 255, 255, 255], 
         [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]]
        ))
        
    def test_blah(self):
        self.assertEqual(1,1)


if __name__ == '__main__':
    unittest.main()

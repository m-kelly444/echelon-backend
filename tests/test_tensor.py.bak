import unittest
import numpy as np
import torch
import echelon_ml as eml

class TestTensor(unittest.TestCase):
    def test_creation(self):
        t1 = eml.tensor([1, 2, 3, 4])
        self.assertEqual(t1.shape, (4,))
        self.assertEqual(t1.size, 4)
        
        t2 = eml.tensor(np.array([[1, 2], [3, 4]]))
        self.assertEqual(t2.shape, (2, 2))
        self.assertEqual(t2.size, 4)
    
    def test_operations(self):
        a = eml.tensor([1, 2, 3])
        b = eml.tensor([4, 5, 6])
        
        c = a + b
        np.testing.assert_array_equal(c.data, np.array([5, 7, 9]))
        
        d = a * b
        np.testing.assert_array_equal(d.data, np.array([4, 10, 18]))
        
        e = a - b
        np.testing.assert_array_equal(e.data, np.array([-3, -3, -3]))
        
        f = a / b
        np.testing.assert_array_almost_equal(f.data, np.array([0.25, 0.4, 0.5]))

if __name__ == '__main__':
    unittest.main()

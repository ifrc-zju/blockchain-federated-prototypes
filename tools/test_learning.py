import json
import unittest
import numpy as np
from learning import init_model, objective, prototype_similarity, shrink_prototypes, wpc_loss

class MechanismTests(unittest.TestCase):
    def test_wpc_gradient_matches_numerical_difference(self):
        rng = np.random.default_rng(77)
        z = rng.normal(size=(5, 4))
        y = np.array([0, 0, 1, 1, 2])
        s = prototype_similarity(rng.normal(size=(3, 4)))
        loss, grad = wpc_loss(z, y, s, .5, .15)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                a, b = z.copy(), z.copy()
                a[i,j] += 1e-6; b[i,j] -= 1e-6
                numeric = (wpc_loss(a,y,s,.5,.15)[0]-wpc_loss(b,y,s,.5,.15)[0])/2e-6
                self.assertAlmostEqual(float(grad[i,j]),float(numeric),places=6)
        self.assertTrue(np.isfinite(loss))

    def test_no_valid_anchor_contributes_zero(self):
        loss, grad = wpc_loss(np.eye(3),np.arange(3),np.eye(3),0,.15)
        self.assertEqual(loss,0)
        np.testing.assert_array_equal(grad,0)

    def test_shrink_zero_count_never_fabricates_observation(self):
        raw = np.array([[2.,4.],[99.,99.],[1.,1.]])
        prior = np.zeros_like(raw)
        counts = np.array([1,0,10])
        out = shrink_prototypes(raw,counts,prior,10)
        np.testing.assert_allclose(out[0],raw[0]/11)
        np.testing.assert_array_equal(out[1],0)
        np.testing.assert_allclose(out[2],raw[2]/2)
        np.testing.assert_array_equal(shrink_prototypes(raw,counts,prior,0)[counts>0],raw[counts>0])

    def test_full_network_gradient(self):
        rng = np.random.default_rng(88)
        model = init_model(5,6,4,3,rng,dtype=np.float64)
        x=rng.random((5,5)); y=np.array([0,0,1,1,2])
        s=prototype_similarity(rng.normal(size=(3,4)))
        loss,grad,_=objective(model,x,y,s,.7,.5,.15,1e-5)
        for key in model:
            for flat in rng.choice(model[key].size,min(4,model[key].size),replace=False):
                idx=np.unravel_index(flat,model[key].shape)
                old=model[key][idx]
                model[key][idx]=old+1e-6
                plus=objective(model,x,y,s,.7,.5,.15,1e-5)[0]
                model[key][idx]=old-1e-6
                minus=objective(model,x,y,s,.7,.5,.15,1e-5)[0]
                model[key][idx]=old
                self.assertAlmostEqual(float(grad[key][idx]),float((plus-minus)/2e-6),places=6)

if __name__ == '__main__':
    unittest.main(verbosity=2)

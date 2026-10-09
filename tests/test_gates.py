import unittest

from models.gates import DomainRouter


class FakeModel:
    def __init__(self):
        self.active_domain = None
        self.losses = {"math": 0.2, "code": 0.8, "hybrid": 0.4}

    def domain_nll(self, prompts):
        return self.losses[self.active_domain]


class DomainRouterTests(unittest.TestCase):
    def test_lower_nll_receives_higher_routing_weight(self):
        model = FakeModel()
        router = DomainRouter(model, ["math", "code", "hybrid"], lambda domain: setattr(model, "active_domain", domain))
        weights = router.fit(["prompt"], verbose=False)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertGreater(weights[0], weights[2])
        self.assertGreater(weights[2], weights[1])
        self.assertEqual(set(router.domain_nlls), {"math", "code", "hybrid"})

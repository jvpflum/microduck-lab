"""Numerical interface tests; CPU tests never optimize or simulate the robot."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec=importlib.util.spec_from_file_location('fresh_training',Path(__file__).with_name('train.py'))
training=importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


class BoundedPhaseTests(unittest.TestCase):
    def test_serious_run_transition_budget(self):
        n,iterations,budget,interval=training.phase_configuration('train')
        self.assertEqual(n*iterations*training.HORIZON,294_912_000)
        self.assertEqual((budget,interval),(7200,250))

    def test_no_over_budget_or_smaller_required_smoke(self):
        for args in [('smoke',32,5,None,None),('train',8192,None,None,None),
                     ('train',None,None,7201,None),('capacity',512,2,None,None),
                     ('smoke',64,6,None,None),('train',4096,0,None,None)]:
            with self.assertRaises(ValueError):training.phase_configuration(*args)


@unittest.skipUnless(importlib.util.find_spec('torch')and importlib.util.find_spec('rsl_rl'),
                     'Pinned Torch/RSL runtime required for numerical model tests')
class FreshActorInterfaceTests(unittest.TestCase):
    def setUp(self):
        import torch
        from tensordict import TensorDict
        self.torch=torch
        self.TensorDict=TensorDict
        torch.manual_seed(19101)
        actor=torch.randn(16,61);actor[:,49:]=0.
        self.obs=TensorDict({'actor':actor,'critic':torch.randn(16,69)},batch_size=[16])

    def test_actor_inputs_mask_only_unused_commands(self):
        x=self.obs['actor'].clone();x[:,49:]=9.
        safe=training.safe_actor_observations(x)
        self.assertTrue(self.torch.equal(safe[:,:49],x[:,:49]))
        self.assertTrue(self.torch.equal(safe[:,34:48],x[:,34:48]))
        self.assertEqual(float(safe[:,49:].abs().max()),0.)
        self.assertEqual(float(x[:,49:].min()),9.)

    def test_critic_privileges_cannot_affect_actor_output(self):
        actor,critic=training.create_models(self.obs,'cpu')
        first=actor(self.obs)
        changed=self.obs.clone();changed['critic']+=1000.
        self.assertTrue(self.torch.equal(first,actor(changed)))
        self.assertFalse(self.torch.equal(critic(self.obs),critic(changed)))

    def test_export_contains_actual_nonidentity_normalizer(self):
        actor,_=training.create_models(self.obs,'cpu')
        # Deliberately train the STATISTICS, never the network. This catches the
        # dangerous export of actor.mlp alone that omits empirical normalization.
        observed=self.obs.clone();observed['actor'][:,:49]*=3.
        observed['actor'][:,:49]+=2.
        with self.torch.inference_mode():actor.update_normalization(observed)
        samples=self.obs['actor'][:8].clone()
        with self.torch.no_grad():
            normalized=training.actor_mean(actor,samples)
            raw=actor.mlp(samples)
        self.assertGreater(float((normalized-raw).abs().max()),1e-3)
        interface={'joint_names':[f'joint{i}'for i in range(14)],'default_joint_pos':[0.]*14,
            'action_scale':[1.]*14,'joint_stiffness':[1.]*14,'joint_damping':[0.]*14,
            'command_names':['twist'],'observation_names':['arena61']}
        with tempfile.TemporaryDirectory()as tmp:
            report=training.export_policy(actor,Path(tmp)/'fresh.onnx',interface,samples)
        self.assertTrue(report['normalizer_baked'])
        self.assertLessEqual(report['normalized_cuda_actor_vs_onnx_max_abs'],training.EXPORT_ATOL)

    def test_fresh_network_has_no_base_or_teacher(self):
        actor,_=training.create_models(self.obs,'cpu')
        self.assertTrue(actor.obs_normalization)
        names=list(actor.state_dict())
        self.assertFalse(any('base' in name or 'teacher' in name for name in names))
        self.assertGreater(sum(p.numel()for p in actor.mlp.parameters()),100_000)
        self.assertEqual(actor.distribution.std_type,'log')


if __name__=='__main__':unittest.main()

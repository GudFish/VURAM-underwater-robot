"""Learner-compatible vehicle, allocation, reward and episode contracts."""
import sys
from pathlib import Path

import numpy as np


class TrainingContract:
    """Import the learner's NumPy/HDF5 contracts without running a simulator."""

    def __init__(self, learning_root):
        root = Path(learning_root).expanduser().resolve()
        if not (root / 'auv_learning').is_dir():
            raise ValueError('learning_root must contain auv_learning: {}'.format(root))
        sys.path.insert(0, str(root))
        import auv_learning
        if Path(auv_learning.__file__).resolve().parents[1] != root:
            raise ValueError('auv_learning was already imported from a different learning_root')
        from auv_learning.deployment.episode import EpisodeBundle
        from auv_learning.dynamics.allocation import WrenchAllocator
        from auv_learning.simulation.environment import reward_fn
        from auv_learning.task.math_utils import load_yaml
        self.bundle_type = EpisodeBundle
        self.reward = reward_fn
        self.env_config = load_yaml(str(root / 'config/env_config.yaml'))
        self.reward_weights = load_yaml(str(root / 'config/task_config.yaml'))['reward_weights']
        self.allocator = WrenchAllocator.from_config(self.env_config['vehicle']['com'],
                                                    self.env_config['thruster'])

    def validate_policy(self, policy):
        """A reload must preserve the meaning of normalized actions/motor state."""
        manifest, allocator = policy.manifest, self.allocator
        expected = dict(action_scales=allocator.scales,
                        negative_action_scales=allocator.negative_scales,
                        thruster_force_range_N=[allocator.min_thrust, allocator.max_forward_thrust])
        expected_model = dict(allocator.curve.as_dict(), tau=allocator.lag_tau)
        actual_model = manifest.get('thruster_model', {})
        for name, value in expected_model.items():
            if isinstance(value, str):
                if actual_model.get(name) != value:
                    raise ValueError('snapshot thruster_model.{} differs from training config'.format(name))
            else:
                expected['thruster_model.' + name] = value
        for name, value in expected.items():
            actual = (actual_model.get(name.split('.', 1)[1])
                      if name.startswith('thruster_model.') else manifest.get(name))
            if actual is None or np.shape(actual) != np.shape(value) or not np.allclose(actual, value):
                raise ValueError('snapshot {} differs from training config'.format(name))


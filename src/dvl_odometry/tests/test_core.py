import math

import numpy as np
import pytest

from dvl_odometry.core import DvlPredictor, RateLimiter, covariance, frd_to_flu
from dvl_odometry.simulation import trajectory


def predictor(velocity=(1., 0., 0.)):
    state = DvlPredictor(vel_timeout=1., dr_timeout=1., growth=0.)
    state.anchor(0., [0., 0., 0.], [0., 0., 0., 1.], np.eye(6)*0.01)
    state.set_velocity(0., velocity, np.eye(3)*0.04)
    return state


def test_frd_to_flu_covariance():
    raw = np.array([[1., .2, .3], [.2, 2., .4], [.3, .4, 3.]])
    v, cov = frd_to_flu([1., 2., 3.], raw)
    np.testing.assert_allclose(v, [1., -2., -3.])
    np.testing.assert_allclose(cov, [[1., -.2, -.3], [-.2, 2., .4], [-.3, .4, 3.]])


@pytest.mark.parametrize('raw', [None, [], [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                      [[float('nan')]*3]*3])
def test_broken_covariance_is_not_perfect(raw):
    assert np.diag(covariance(raw)).min() > 0.


def test_rate_limit_is_a_cap_not_a_resampler():
    cap = RateLimiter(5.)
    assert [t for t in (0., .05, .1, .2, .25, .6) if cap.allow(t)] == [0., .2, .6]
    cap = RateLimiter(20.)
    assert all(cap.allow(t) for t in (0., .5, 1., 1.5))  # a 2-Hz sensor remains 2 Hz
    assert RateLimiter(0.).allow(0.)
    with pytest.raises(ValueError):
        RateLimiter(-1.)


def test_new_velocity_does_not_rewrite_the_previous_interval():
    state = predictor()
    state.set_velocity(.4, [2., 0., 0.], np.eye(3)*.04)
    state.advance(.6)
    assert state.position[0] == pytest.approx(.8)


def test_timeout_clips_integration_and_holds_last_predicted_position():
    state = predictor()
    state.advance(.8)
    state.advance(2.)
    assert state.position[0] == pytest.approx(1.)
    state.advance(3.)
    assert state.position[0] == pytest.approx(1.)
    assert not state.predicting(3.)


def test_invalid_velocity_does_not_snap_back_to_anchor():
    state = predictor()
    state.invalidate_velocity(.3)
    state.advance(.8)
    assert state.position[0] == pytest.approx(.3)
    state.set_velocity(.8, [2., 0., 0.], np.eye(3)*.04)
    state.advance(.9)
    assert state.position[0] == pytest.approx(.5)  # no integration over the missing interval


def test_fresh_velocity_cannot_extrapolate_an_expired_pose():
    state = predictor()
    state.set_velocity(.8, [1., 0., 0.], np.eye(3)*.04)
    state.advance(2.)
    assert state.position[0] == pytest.approx(1.)


def test_covariance_does_not_depend_on_timer_frequency():
    slow, fast = predictor(), predictor()
    slow.advance(.8)
    for stamp in np.arange(.05, .81, .05):
        fast.advance(stamp)
    np.testing.assert_allclose(slow.position, fast.position)
    np.testing.assert_allclose(slow.pose_cov, fast.pose_cov)


def test_pose_rotation_and_anchor_correction():
    state = predictor()
    state.anchor(0., [0, 0, 0], [0, 0, math.sin(math.pi/4), math.cos(math.pi/4)], np.eye(6)*.01)
    state.advance(.5)
    np.testing.assert_allclose(state.position, [0., .5, 0.], atol=1e-12)
    state.anchor(.6, [2, 3, 4], [0, 0, 0, 1], np.eye(6)*.01)
    np.testing.assert_allclose(state.position, [2, 3, 4])
    state.clear()
    assert state.position is None and not state.vel_valid


def test_turning_simulator_reports_body_velocity_not_world_velocity():
    _, _, body_velocity, _, _ = trajectory(3., speed=.3, yaw_rate=.5)
    np.testing.assert_allclose(body_velocity, [.3, 0., 0.], atol=1e-12)


@pytest.mark.parametrize('roll,pitch', [(0., 0.), (.2, -.3)])
def test_stationary_specific_force_gravity_compensation(roll, pitch):
    _, rpy, velocity, force, omega = trajectory(2., speed=0., yaw_rate=0., roll=roll, pitch=pitch)
    from tf.transformations import quaternion_from_euler
    from dvl_odometry.core import rotation
    rot = rotation(quaternion_from_euler(*rpy))
    np.testing.assert_allclose(rot.dot(force)-[0., 0., 9.80665], [0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(velocity, 0.)
    np.testing.assert_allclose(omega, 0.)


def test_simulator_reset_zeroes_full_attitude_without_changing_body_velocity():
    from dvl_odometry.simulator import DvlSimulator
    sim = DvlSimulator()
    state = trajectory(3., yaw_rate=.2, roll=.15, pitch=-.25)
    sim.state = lambda: state
    class Sink:
        def sendall(self, _data):
            pass
    before = sim.velocity_report()
    sim.command(Sink(), {'command': 'reset_dead_reckoning'})
    report = sim.dr_report()
    np.testing.assert_allclose([report[k] for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')],
                               np.zeros(6), atol=1e-12)
    after = sim.velocity_report()
    np.testing.assert_allclose([after[k] for k in ('vx', 'vy', 'vz')],
                               [before[k] for k in ('vx', 'vy', 'vz')])

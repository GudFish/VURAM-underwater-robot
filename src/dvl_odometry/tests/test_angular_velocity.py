import math

import numpy as np
import pytest

from dvl_odometry.angular_velocity import AngularVelocityEstimator
from dvl_odometry.core import UNKNOWN_VARIANCE, rotation


def quaternion(vector):
    vector = np.asarray(vector, dtype=float)
    angle = np.linalg.norm(vector)
    if angle < 1e-15:
        return np.array([0., 0., 0., 1.])
    return np.r_[vector * (math.sin(angle / 2.) / angle), math.cos(angle / 2.)]


def multiply(first, second):
    av, aw = first[:3], first[3]
    bv, bw = second[:3], second[3]
    return np.r_[aw * bv + bw * av + np.cross(av, bv), aw * bw - av.dot(bv)]


def assert_unknown(estimator, stamp):
    omega, cov, valid = estimator.estimate(stamp)
    assert not valid
    np.testing.assert_array_equal(omega, np.zeros(3))
    np.testing.assert_array_equal(cov, np.eye(3) * UNKNOWN_VARIANCE)


def test_first_sample_and_reset_are_unknown_not_measured_zero():
    estimator = AngularVelocityEstimator()
    assert_unknown(estimator, 0.)
    assert not estimator.update(0., quaternion([0, 0, 0]))
    assert_unknown(estimator, 0.)
    assert estimator.update(.1, quaternion([0, 0, .02]))
    estimator.reset()
    assert_unknown(estimator, .1)


@pytest.mark.parametrize('omega', [(0.2, 0., 0.), (0., -.3, 0.), (0., 0., .4),
                                  (.2, -.35, .4), (1e-10, 2e-10, -1e-10)])
def test_body_rates_on_all_axes_at_arbitrary_initial_attitude(omega):
    estimator = AngularVelocityEstimator(filter_tau=0.)
    omega = np.array(omega)
    initial = multiply(quaternion([.4, 0, 0]), quaternion([0, -.7, .3]))
    estimator.update(5., initial)
    current = multiply(initial, quaternion(omega * .13))
    assert estimator.update(5.13, current)
    actual, cov, valid = estimator.estimate(5.13)
    assert valid
    np.testing.assert_allclose(actual, omega, rtol=1e-10, atol=1e-13)
    assert np.linalg.eigvalsh(cov).min() >= .05 ** 2 - 1e-12


def test_quaternion_sign_flips_do_not_create_angular_motion():
    estimator = AngularVelocityEstimator(filter_tau=0.)
    pose = quaternion([.4, -.3, .7])
    estimator.update(0., pose)
    assert estimator.update(.1, -pose)
    np.testing.assert_allclose(estimator.estimate(.1)[0], 0., atol=1e-13)
    assert estimator.update(.2, pose)
    np.testing.assert_allclose(estimator.estimate(.2)[0], 0., atol=1e-13)


def test_yaw_wrap_at_pi_uses_the_short_rotation():
    estimator = AngularVelocityEstimator(filter_tau=0.)
    estimator.update(0., quaternion([0, 0, math.pi - .02]))
    assert estimator.update(.1, quaternion([0, 0, -math.pi + .02]))
    np.testing.assert_allclose(estimator.estimate(.1)[0], [0, 0, .4], atol=1e-12)


def test_exact_half_turn_remains_finite_and_sign_invariant():
    for sign in (1, -1):
        estimator = AngularVelocityEstimator(filter_tau=0., max_rate=10., max_angle=math.pi)
        estimator.update(0., quaternion([0, 0, 0]))
        assert estimator.update(.5, sign * np.array([1., 0, 0, 0]))
        omega, cov, valid = estimator.estimate(.5)
        assert valid and np.isfinite(cov).all()
        np.testing.assert_allclose(omega, [2. * math.pi, 0, 0], atol=1e-12)


def test_actual_irregular_intervals_control_differentiation():
    estimator = AngularVelocityEstimator(filter_tau=0.)
    omega = np.array([.1, .2, -.3])
    estimator.update(0., quaternion([0, 0, 0]))
    for stamp in (.08, .23, .51, .93):
        assert estimator.update(stamp, quaternion(omega * stamp))
        np.testing.assert_allclose(estimator.estimate(stamp)[0], omega, atol=1e-12)


def test_fixed_roll_pitch_world_yaw_is_not_a_pure_body_z_rate():
    estimator = AngularVelocityEstimator(filter_tau=0.)
    roll_pitch = multiply(quaternion([0., -.4, 0.]), quaternion([.3, 0., 0.]))
    estimator.update(0., roll_pitch)
    yaw_rate, dt = -.8, .13
    pose = multiply(quaternion([0., 0., yaw_rate * dt]), roll_pitch)
    assert estimator.update(dt, pose)
    expected = rotation(roll_pitch).T.dot([0., 0., yaw_rate])
    np.testing.assert_allclose(estimator.estimate(dt)[0], expected, atol=1e-12)
    assert abs(expected[0]) > .1 and abs(expected[1]) > .1


def test_filter_uses_elapsed_time_and_current_body_coordinates():
    estimator = AngularVelocityEstimator(filter_tau=.3)
    q0 = quaternion([0, 0, 0])
    q1 = quaternion([0, 0, .1])
    q2 = multiply(q1, quaternion([.2, 0, 0]))
    estimator.update(0., q0)
    estimator.update(.2, q1)
    before, previous_cov, _ = estimator.estimate(.2)
    estimator.update(.4, q2)
    alpha = 1. - math.exp(-.2 / .3)
    old_to_current = rotation(q2).T.dot(rotation(q1))
    expected = (1. - alpha) * old_to_current.dot(before) + alpha * np.array([1., 0, 0])
    actual, cov, valid = estimator.estimate(.4)
    assert valid and np.linalg.eigvalsh(cov).min() > 0
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    # A z rate in the previous body frame gains a y component after body roll.
    assert actual[1] > 0
    assert np.isfinite(previous_cov).all()


def test_covariance_uses_both_attitude_samples_and_actual_dt():
    pose_cov = np.diag([.001, .002, .003])
    estimator = AngularVelocityEstimator(filter_tau=0., min_std=.01)
    estimator.update(0., quaternion([0, 0, 0]), pose_cov)
    estimator.update(.2, quaternion([0, 0, 0]), pose_cov * 2)
    # core covariance sanitization adds 1e-10 to each attitude diagonal.
    expected = (pose_cov * 3 + np.eye(3) * 2e-10) / .2 ** 2
    np.testing.assert_allclose(estimator.estimate(.2)[1], expected)


@pytest.mark.parametrize('bad_cov', [None, [], np.full((3, 3), np.nan), -np.eye(3)])
def test_missing_or_broken_covariance_does_not_claim_perfect_rates(bad_cov):
    estimator = AngularVelocityEstimator()
    estimator.update(0., quaternion([0, 0, 0]), bad_cov)
    estimator.update(.1, quaternion([0, 0, .02]), bad_cov)
    assert np.diag(estimator.estimate(.1)[1]).min() >= .05 ** 2


@pytest.mark.parametrize('stamp', [0., -.1, 1.01])
def test_duplicate_backward_and_long_gaps_invalidate_then_reanchor(stamp):
    estimator = AngularVelocityEstimator(timeout=.5, filter_tau=0.)
    estimator.update(0., quaternion([0, 0, 0]))
    estimator.update(.1, quaternion([0, 0, .01]))
    assert not estimator.update(stamp, quaternion([.3, -.2, .4]))
    assert_unknown(estimator, stamp)
    current = multiply(quaternion([.3, -.2, .4]), quaternion([.01, 0, 0]))
    assert estimator.update(stamp + .1, current)
    np.testing.assert_allclose(estimator.estimate(stamp + .1)[0], [.1, 0, 0], atol=1e-12)


@pytest.mark.parametrize('options, angle, dt', [({'max_rate': .5}, .1, .1),
                                              ({'max_angle': .05}, .1, .5)])
def test_jump_rejected_instead_of_clipped_and_next_pose_reanchors(options, angle, dt):
    estimator = AngularVelocityEstimator(filter_tau=0., **options)
    estimator.update(0., quaternion([0, 0, 0]))
    assert not estimator.update(dt, quaternion([0, 0, angle]))
    assert_unknown(estimator, dt)
    assert estimator.update(dt + .1, quaternion([0, 0, angle + .01]))
    np.testing.assert_allclose(estimator.estimate(dt + .1)[0], [0, 0, .1], atol=1e-12)


def test_stale_future_and_nonfinite_queries_are_unknown():
    estimator = AngularVelocityEstimator(timeout=.5)
    estimator.update(0., quaternion([0, 0, 0]))
    estimator.update(.1, quaternion([0, 0, .01]))
    assert estimator.estimate(.6)[2]
    for stamp in (.60001, .09, float('nan'), None):
        assert_unknown(estimator, stamp)


@pytest.mark.parametrize('stamp, pose', [(float('nan'), [0, 0, 0, 1]),
                                        (.2, [0, 0, 0, 0]), (.2, [0, 0, 1]),
                                        (.2, [0, float('inf'), 0, 1])])
def test_invalid_measurements_clear_previous_estimates(stamp, pose):
    estimator = AngularVelocityEstimator()
    estimator.update(0., quaternion([0, 0, 0]))
    estimator.update(.1, quaternion([0, 0, .01]))
    assert not estimator.update(stamp, pose)
    assert_unknown(estimator, .2)
    assert not estimator.update(.3, quaternion([0, 0, .03]))


def test_results_are_copies_not_mutable_internal_state():
    estimator = AngularVelocityEstimator()
    estimator.update(0., quaternion([0, 0, 0]))
    estimator.update(.1, quaternion([0, 0, .01]))
    omega, cov, _ = estimator.estimate(.1)
    omega[:] = 123
    cov[:] = 123
    np.testing.assert_allclose(estimator.estimate(.1)[0], [0, 0, .1], atol=1e-12)
    assert (estimator.estimate(.1)[1] != 123).all()


@pytest.mark.parametrize('options', [{'timeout': 0}, {'filter_tau': -1},
                                   {'max_rate': 0}, {'max_angle': 0},
                                   {'max_angle': math.pi + .01}, {'min_std': 0},
                                   {'timeout': float('inf')}])
def test_invalid_configuration_rejected(options):
    with pytest.raises(ValueError):
        AngularVelocityEstimator(**options)

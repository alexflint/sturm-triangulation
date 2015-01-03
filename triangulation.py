import collections
import numpy as np

import utils


Pose = collections.namedtuple('Pose', ['position', 'orientation'])


def landmark_from_depth(observation, pose, depth):
    f = utils.normalized(utils.unpr(observation))
    return np.dot(pose.orientation.T, f * depth) + pose.position


def householder(x):
    assert len(x) == 3, 'shape was %s' % np.shape(x)
    a = (np.arange(3) == np.argmax(np.abs(x))).astype(float)
    u = utils.normalized(np.cross(x, a))
    v = utils.normalized(np.cross(x, u))
    return np.array([u, v])


def triangulate_depth_householder(observations, poses, base_observation, base_pose):
    f = base_observation
    if len(f) == 2:
        f = utils.normalized(utils.unpr(f))
    assert len(f) == 3, 'f was '+str(f)
    assert len(observations) == len(poses)

    m, c = 0., 0.
    for pose, z in zip(poses, observations):
        h = householder(utils.unpr(z))
        a = utils.dots(h, pose.orientation, base_pose.orientation.T)
        pdiff = base_pose.position - pose.position
        m += float(utils.dots(f, a.T, h, pose.orientation, pdiff))
        c += float(utils.dots(f, a.T, a, f))
    return -m/c


def project(pose, point):
    return utils.pr(np.dot(pose.orientation, point - pose.position))


def reprojection_cost(depth, observation, pose, base_observation, base_pose):
    x = landmark_from_depth(base_observation, base_pose, depth)
    r = project(pose, x) - observation
    return np.dot(r, r)


def sum_reprojection_cost(depth, observations, poses, base_observation, base_pose):
    return sum(reprojection_cost(depth, observation, pose, base_observation, base_pose)
               for observation, pose in zip(observations, poses))


def flatdot(a, b):
    return [sum(float(aij)*bi for aij, bi in zip(ai, b)) for ai in a]


def flatsub(a, b):
    return [ai - bi for ai, bi in zip(a, b)]


def reprojection_cost_derivative_poly(observation, pose, base_observation, base_pose):
    assert len(observation) == 2

    d = np.poly1d([1., 0.])
    f = utils.normalized(utils.unpr(base_observation))
    x = [d * float(p) for p in np.dot(base_pose.orientation.T, f) + base_pose.position]

    y = flatdot(pose.orientation, flatsub(x, pose.position))
    z = map(float, observation)

    a = map(float, flatdot(pose.orientation, flatdot(base_pose.orientation.T, f)))
    b = flatdot(pose.orientation, base_pose.position - pose.position)

    top = (a[0]*y[2]*(y[0]-y[2]*z[0]) +
           a[1]*y[2]*(y[1]-y[2]*z[1]) +
           a[2]*(z[0]*y[0]*y[2] + z[1]*y[1]*y[2] - y[0]*y[0] - y[1]*y[1])) * 2.
    bottom = y[2]*y[2]*y[2]

    return top, bottom


def reconstruction_error(estimated_depth, base_observation, base_pose, true_point):
    estimated_point = landmark_from_depth(base_observation, base_pose, estimated_depth)
    return np.linalg.norm(estimated_point - true_point)


def triangulate_depth_sturm_two_views(observation, pose, base_observation, base_pose):
    top, bottom = reprojection_cost_derivative_poly(observation, pose, base_observation, base_pose)
    roots = np.roots(top)

    best_cost = None
    best_depth = None
    for root in roots:
        if abs(root.imag) < 1e-8:
            cost = reprojection_cost(root.real, observation, pose, base_observation, base_pose)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_depth = root.real

    return best_depth


def triangulate_depth_sturm(observations, poses, base_observation, base_pose):
    assert len(observations) > 0

    quotients = [reprojection_cost_derivative_poly(observation, pose, base_observation, base_pose)
                 for observation, pose in zip(observations, poses)]

    f = np.poly1d([])
    for i in range(len(quotients)):
        term = quotients[i][0]
        for j in range(len(quotients)):
            if i != j:
                term *= quotients[j][1]
        f += term

    print 'Solving polynomial of degree %d' % len(f)
    roots = np.roots(f)

    best_cost = None
    best_depth = None
    for root in roots:
        if abs(root.imag) < 1e-8:
            cost = sum_reprojection_cost(root.real, observations, poses, base_observation, base_pose)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_depth = root.real

    return best_depth

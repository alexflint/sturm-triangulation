import os
import itertools
import collections
import numdifftools
import numpy as np

import utils
import rotation
import triangulation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


Observation = collections.namedtuple('Observation', ['frame_index', 'point_index', 'feature'])


Camera = collections.namedtuple('Camera', ['intrinsics', 'pose'])


Bundle = collections.namedtuple('Bundle', ['cameras', 'points', 'tracks'])


def load_vgg_cameras(basename):
    cameras = []
    for index in itertools.count():
        path = '%s.%03d.P' % (basename, index)
        try:
            pose = np.loadtxt(path)
        except IOError:
            return cameras
        k, r, p = krp_from_pose(pose)
        cameras.append(Camera(intrinsics=k, pose=triangulation.Pose(orientation=r, position=p)))


def load_vgg_points(basename):
    return np.loadtxt(basename + '.p3d')


def load_vgg_corners(basename):
    corners = []
    for index in itertools.count():
        path = '%s.%03d.corners' % (basename, index)
        try:
            corners.append(np.loadtxt(path))
        except IOError:
            return corners


def load_vgg_tracks(basename):
    corners = load_vgg_corners(basename)
    tracks = []
    with open(basename+'.nview-corners') as fd:
        for point_index, line in enumerate(fd):
            track = []
            tokens = line.split()
            assert len(tokens) == len(corners)
            for frame_index, corner_index in enumerate(tokens):
                if corner_index != '*':
                    track.append(Observation(frame_index=frame_index,
                                             point_index=point_index,
                                             feature=corners[frame_index][int(corner_index)]))
            tracks.append(track)
    return tracks


def load_vgg_dataset(basename):
    return Bundle(cameras=load_vgg_cameras(basename),
                  points=load_vgg_points(basename),
                  tracks=load_vgg_tracks(basename))


def load_matlab_dataset(basename):
    import scipy.io
    poses = scipy.io.loadmat(basename+"_Ps.mat")['P'][0]
    features = np.loadtxt(basename+"_tracks.xy")

    cameras = []
    for pose in poses:
        k, r, p = krp_from_pose(pose)
        cameras.append(Camera(intrinsics=k, pose=triangulation.Pose(orientation=r, position=p)))

    tracks = []
    for i, row in enumerate(features):
        track = []
        for j, feature in enumerate(row.reshape((-1, 2))):
            if not np.all(feature == (-1, -1)):
                track.append(Observation(frame_index=j, point_index=i, feature=feature))
        tracks.append(track)

    return Bundle(cameras=cameras, points=None, tracks=tracks)


def krp_from_pose(pose):
    assert pose.shape == (3, 4)
    qq, rr = np.linalg.qr(pose[:,:3].T)
    k = rr.T
    r = qq.T
    p = -np.dot(r.T, np.linalg.solve(k, pose[:, 3]))
    return k, r, p


def bundle_reprojection_error(bundle, observation):
    pt = bundle.points[observation.point_index]
    cam = bundle.cameras[observation.frame_index]
    z = utils.pr(np.dot(cam.intrinsics, np.dot(cam.pose.orientation, pt - cam.pose.position)))
    return z - observation.feature


def feature_to_calibrated(k, f):
    return utils.pr(np.linalg.solve(k, utils.unpr(f)))


def run_from_dataset():
    #bundle = load_vgg_dataset('data/corridor/bt')
    bundle = load_matlab_dataset('data/dinosaur/dino')

    errors_householder = []
    errors_sturm = []

    points_householder = []
    points_sturm = []

    for i, track in enumerate(bundle.tracks):
        observations = []
        poses = []
        for observation in track:
            cam = bundle.cameras[observation.frame_index]
            observations.append(feature_to_calibrated(cam.intrinsics, observation.feature))
            poses.append(cam.pose)

        base_pose = poses.pop()
        base_observation = observations.pop()

        estimated_depth_householder = triangulation.triangulate_depth_householder(
            observations, poses, base_observation, base_pose)

        estimated_depth_sturm = triangulation.triangulate_depth_sturm(
            observations, poses, base_observation, base_pose)

        if estimated_depth_sturm is None:
            print 'Warning: sturm triangulation returned None'
            continue

        reconstruction_householder = triangulation.landmark_from_depth(base_observation, base_pose, estimated_depth_householder)
        reconstruction_sturm = triangulation.landmark_from_depth(base_observation, base_pose, estimated_depth_sturm)

        points_householder.append(reconstruction_householder)
        points_sturm.append(reconstruction_sturm)

        if bundle.points is not None:
            true_point = bundle.points[i]
            errors_householder.append(np.linalg.norm(reconstruction_householder - true_point))
            errors_sturm.append(np.linalg.norm(reconstruction_sturm - true_point))

    np.savetxt('out/points_householder.txt', points_householder)
    np.savetxt('out/points_sturm.txt', points_sturm)

    if bundle.points is not None:
        plt.clf()
        plt.hist(np.log10(errors_householder), bins=30, normed=True, label='Householder', alpha=.4)
        plt.hist(np.log10(errors_sturm), bins=30, normed=True, label='Sturm', alpha=.5)
        plt.xlabel('Log10 reconstruction error')
        plt.legend()
        plt.savefig('out/corridor_errors.pdf')


def run_test():
    np.random.seed(0)
    pose = triangulation.Pose(orientation=rotation.exp(np.random.randn(3)), position=np.random.randn(3))
    base_pose = triangulation.Pose(orientation=np.eye(3), position=np.zeros(3))

    point = np.random.randn(3) * 10 + 20
    observation = triangulation.project(pose, point)
    base_observation = triangulation.project(base_pose, point)

    true_depth = np.linalg.norm(np.dot(base_pose.orientation, point - base_pose.position))

    estimated_depth = triangulation.triangulate_depth_sturm_two_views(observation, pose, base_observation, base_pose)
    estimated_depth2 = triangulation.triangulate_depth_sturm([observation], [pose], base_observation, base_pose)

    print 'True depth:', true_depth
    print 'Estimated depth:', estimated_depth
    print 'Estimated depth 2:', estimated_depth2
    return

    eval_depth = 2.

    f = lambda d: triangulation.reprojection_cost(d, observation, pose, base_observation, base_pose)
    numerical_derivative = np.squeeze(numdifftools.Derivative(f)(eval_depth))
    analytic_derivative = reprojection_cost_derivative(eval_depth, observation, pose, base_observation, base_pose)

    print 'Numerical:', numerical_derivative
    print 'Analytic:', analytic_derivative

    alt_derivative = reprojection_cost_derivative_alt(eval_depth, observation, pose, base_observation, base_pose)
    print 'Alt:', alt_derivative


def run_simulation():
    np.random.seed(0)

    num_frames = 4
    num_experiments = 2000

    noise = 1e-2

    base_pose = triangulation.Pose(orientation=np.eye(3), position=np.zeros(3))

    errors_householder = []
    errors_sturm = []

    for i in range(num_experiments):
        true_point = np.random.randn(3) * 10
        positions = np.random.randn(num_frames, 3)
        axisangles = np.random.randn(num_frames, 3) * .1
        poses = [triangulation.Pose(orientation=rotation.exp(w), position=p)
                 for w, p in zip(axisangles, positions)]

        true_base_observation = triangulation.project(base_pose, true_point)
        true_observations = np.array([triangulation.project(pose, true_point) for pose in poses])

        observations = true_observations + np.random.randn(*true_observations.shape) * noise

        estimated_depth_householder = triangulation.triangulate_depth_householder(
            observations, poses, true_base_observation, base_pose)

        estimated_depth_sturm = triangulation.triangulate_depth_sturm(
            observations, poses, true_base_observation, base_pose)

        errors_householder.append(triangulation.reconstruction_error(estimated_depth_householder,
                                                                           true_base_observation,
                                                                           base_pose,
                                                                           true_point))

        errors_sturm.append(triangulation.reconstruction_error(estimated_depth_sturm,
                                                                     true_base_observation,
                                                                     base_pose,
                                                                     true_point))

    plt.clf()
    plt.hist(np.log10(errors_householder), bins=30, normed=True, label='Householder', alpha=.4)
    plt.hist(np.log10(errors_sturm), bins=30, normed=True, label='Sturm', alpha=.5)
    plt.xlabel('Log10 reconstruction error')
    plt.legend()
    plt.savefig('out/errors.pdf')

if __name__ == '__main__':
    #run_test()
    #run_simulation()
    run_from_dataset()


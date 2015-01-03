import numdifftools
import numpy as np

import utils
import rotation
import triangulation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


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

    point_errors_householder = []
    point_errors_sturm = []

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

        point_errors_householder.append(triangulation.reconstruction_error(estimated_depth_householder,
                                                                           true_base_observation,
                                                                           base_pose,
                                                                           true_point))

        point_errors_sturm.append(triangulation.reconstruction_error(estimated_depth_sturm,
                                                                     true_base_observation,
                                                                     base_pose,
                                                                     true_point))

    plt.clf()
    plt.hist(np.log10(point_errors_householder), bins=30, normed=True, label='Householder', alpha=.4)
    plt.hist(np.log10(point_errors_sturm), bins=30, normed=True, label='Sturm', alpha=.5)
    #plt.ylim(0, 1)
    plt.xlabel('Log10 reconstruction error')
    plt.legend()
    plt.savefig('out/errors.pdf')

if __name__ == '__main__':
    #run_test()
    run_simulation()


#!/usr/bin/env python3

import numpy as np
import rospy
import tf.transformations as tf_trans
from geometry_msgs.msg import PoseArray, PoseStamped
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2

from kiss_slam.pipeline import SlamPipeline


class KissSlamNode:
    def __init__(self):
        cloud_topic = rospy.get_param("~pointcloud_topic", "/points")
        config = rospy.get_param("~config", None)
        if isinstance(config, str) and config.strip() == "":
            config = None
        self.frame_id = rospy.get_param("~frame_id", "map")

        self.pipeline = SlamPipeline(dataset=None, config_file=config)

        self.pose_pub = rospy.Publisher("~pose", PoseStamped, queue_size=1)
        self.poses_pub = rospy.Publisher("~poses", PoseArray, queue_size=10)
        self.sub = rospy.Subscriber(cloud_topic, PointCloud2, self.callback, queue_size=1)

        self.trajectory = []

    def callback(self, msg: PointCloud2):
        points = np.array(
            [
                [p[0], p[1], p[2]]
                for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            ]
        )
        timestamps = np.zeros(points.shape[0])
        self.pipeline.kiss_slam.process_scan(points, timestamps)

        T = self.pipeline.kiss_slam.odometry.last_pose
        self.publish_pose(T, msg.header.stamp)

    def publish_pose(self, T: np.ndarray, stamp):
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = float(T[0, 3])
        pose.pose.position.y = float(T[1, 3])
        pose.pose.position.z = float(T[2, 3])
        q = tf_trans.quaternion_from_matrix(T)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        self.pose_pub.publish(pose)
        self.trajectory.append(pose.pose)

        pa = PoseArray()
        pa.header.stamp = stamp
        pa.header.frame_id = self.frame_id
        pa.poses = self.trajectory
        self.poses_pub.publish(pa)


def main():
    rospy.init_node("kiss_slam_node")
    node = KissSlamNode()
    rospy.spin()


if __name__ == "__main__":
    main()

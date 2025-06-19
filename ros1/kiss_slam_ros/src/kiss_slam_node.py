#!/usr/bin/env python3

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from sensor_msgs import point_cloud2
from std_msgs.msg import Bool
from std_srvs.srv import Trigger, TriggerResponse
import tf.transformations as tf_trans
import open3d as o3d
import itertools

from kiss_slam.config import load_config
from kiss_slam.slam import KissSLAM


class KissSlamNode:
    def __init__(self):
        cloud_topic = rospy.get_param("~pointcloud_topic", "/points")
        config = rospy.get_param("~config", None)
        self.frame_id = rospy.get_param("~frame_id", "map")

        slam_config = load_config(config)
        self.kiss_slam = KissSLAM(slam_config)

        self.pose_pub = rospy.Publisher("~pose", PoseStamped, queue_size=1)
        self.closure_pub = rospy.Publisher("~loop_closure", Bool, queue_size=10)
        self.sub = rospy.Subscriber(cloud_topic, PointCloud2, self.callback, queue_size=1)
        self.save_srv = rospy.Service("~save_map", Trigger, self.handle_save_map)

        self.last_closure_count = 0

    def callback(self, msg: PointCloud2):
        points_iter = point_cloud2.read_points(
            msg, field_names=("x", "y", "z", "t"), skip_nans=True
        )
        points_and_times = np.fromiter(
            itertools.chain.from_iterable(points_iter), dtype=np.float32
        ).reshape(-1, 4)

        points = points_and_times[:, :3]
        timestamps = points_and_times[:, 3]

        self.kiss_slam.process_scan(points, np.empty((0,))) #timestamps provides worse performance

        # Check for new loop closures and notify
        closures = len(self.kiss_slam.get_closures())
        if closures > self.last_closure_count:
            rospy.loginfo("KissSLAM: Loop closure detected")
            self.closure_pub.publish(True)
        else:
            self.closure_pub.publish(False)
        self.last_closure_count = closures

        T = self.kiss_slam.odometry.last_pose
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

    def compute_global_map(self) -> np.ndarray:
        points = []
        for node in self.kiss_slam.local_map_graph.local_maps():
            if node.pcd is None:
                continue
            local_pts = node.pcd.point.positions.numpy()
            pts = local_pts @ node.keypose[:3, :3].T + node.keypose[:3, 3]
            points.append(pts)

        current_pts = self.kiss_slam.voxel_grid.point_cloud()
        if current_pts.size != 0:
            keypose = self.kiss_slam.local_map_graph.last_keypose
            pts = current_pts @ keypose[:3, :3].T + keypose[:3, 3]
            points.append(pts)

        if not points:
            return np.empty((0, 3), np.float32)
        return np.concatenate(points, axis=0)

    def handle_save_map(self, req):
        output = rospy.get_param("~map_output", "/tmp/kiss_slam_map.pcd")
        points = self.compute_global_map()
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        o3d.io.write_point_cloud(output, pcd)
        return TriggerResponse(success=True, message=f"Saved map to {output}")


def main():
    rospy.init_node("kiss_slam_node")
    node = KissSlamNode()
    rospy.spin()


if __name__ == "__main__":
    main()

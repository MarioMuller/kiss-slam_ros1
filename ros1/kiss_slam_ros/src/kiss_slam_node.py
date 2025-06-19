#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import rospy
import tf.transformations as tf_trans
from geometry_msgs.msg import PoseArray, PoseStamped
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Header
from std_srvs.srv import Trigger, TriggerResponse

from kiss_slam.config import load_config
from kiss_slam.slam import KissSLAM


def cloud_to_xyz_t(msg: PointCloud2) -> np.ndarray:
    """Convert ``msg`` into an ``(N, 4)`` ``float32`` array with ``x``/``y``/``z``/``t``.

    The routine bypasses ``read_points`` for performance by directly reading the
    raw buffer. If the cloud does not contain a ``t`` field, the returned column
    is filled with zeros. Big-endian clouds are not supported.
    """

    if msg.is_bigendian:
        raise ValueError("Big-endian PointCloud2 not supported")

    field_map = {f.name: f for f in msg.fields}
    for name in ("x", "y", "z"):
        if name not in field_map:
            raise ValueError(f"Missing required field '{name}'")

    step_floats = msg.point_step // 4
    cloud = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, step_floats)

    idx = [field_map[name].offset // 4 for name in ("x", "y", "z")]
    xyz = cloud[:, idx]

    if "t" in field_map:
        t_idx = field_map["t"].offset // 4
        t = cloud[:, t_idx][:, np.newaxis]
    else:
        t = np.zeros((xyz.shape[0], 1), dtype=np.float32)

    return np.concatenate((xyz, t), axis=1)


class KissSlamNode:
    def __init__(self):
        cloud_topic = rospy.get_param("~pointcloud_topic", "/points")
        config = rospy.get_param("~config", None)
        self.frame_id = rospy.get_param("~frame_id", "map")

        slam_config = load_config(config)
        self.kiss_slam = KissSLAM(slam_config)

        self.pose_pub = rospy.Publisher("~pose", PoseStamped, queue_size=1)
        self.poses_pub = rospy.Publisher("~poses", PoseArray, queue_size=10)
        self.map_pub = rospy.Publisher("~global_map", PointCloud2, queue_size=1, latch=True)
        self.closure_pub = rospy.Publisher("~loop_closure", Bool, queue_size=10)
        self.sub = rospy.Subscriber(cloud_topic, PointCloud2, self.callback, queue_size=1)
        self.save_srv = rospy.Service("~save_map", Trigger, self.handle_save_map)

        self.trajectory = []
        self.last_closure_count = 0

    def callback(self, msg: PointCloud2):
        points_and_times = cloud_to_xyz_t(msg)

        points = points_and_times[:, :3]

        self.kiss_slam.process_scan(points, timestamps)

        # Check for new loop closures and notify
        closures = len(self.kiss_slam.get_closures())
        if closures > self.last_closure_count:
            rospy.loginfo("KissSLAM: Loop closure detected")
            self.closure_pub.publish(True)
        else:
            self.closure_pub.publish(False)
        self.last_closure_count = closures

        # Publish updated global map if a new local map was created
        # self.publish_global_map()

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
        self.trajectory.append(pose.pose)

        pa = PoseArray()
        pa.header.stamp = stamp
        pa.header.frame_id = self.frame_id
        pa.poses = self.trajectory
        self.poses_pub.publish(pa)

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

    def publish_global_map(self):
        points = self.compute_global_map()
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.frame_id
        cloud = point_cloud2.create_cloud_xyz32(header, points)
        self.map_pub.publish(cloud)

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

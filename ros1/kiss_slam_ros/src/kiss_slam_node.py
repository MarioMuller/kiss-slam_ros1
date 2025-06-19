#!/usr/bin/env python3

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
from sensor_msgs import point_cloud2
from std_srvs.srv import Trigger, TriggerResponse
import tf.transformations as tf_trans
import open3d as o3d

from kiss_slam.config import load_config
from kiss_slam.slam import KissSLAM


def pointcloud2_to_numpy(msg: PointCloud2):
    """Convert a ``PointCloud2`` message into numpy arrays."""
    if hasattr(point_cloud2, "read_points_numpy"):
        data = point_cloud2.read_points_numpy(
            msg, field_names=("x", "y", "z", "t"), skip_nans=True
        )
        return data[:, :3], data[:, 3]

    dtype_mappings = {
        PointField.INT8: np.int8,
        PointField.UINT8: np.uint8,
        PointField.INT16: np.int16,
        PointField.UINT16: np.uint16,
        PointField.INT32: np.int32,
        PointField.UINT32: np.uint32,
        PointField.FLOAT32: np.float32,
        PointField.FLOAT64: np.float64,
    }

    byte_order = ">" if msg.is_bigendian else "<"
    dtype_list = []
    offset = 0
    for field in msg.fields:
        while offset < field.offset:
            dtype_list.append((f"_pad{offset}", byte_order + "u1"))
            offset += 1
        base_dtype = np.dtype(dtype_mappings[field.datatype])
        field_dtype = base_dtype.newbyteorder(byte_order)
        if field.count == 1:
            dtype_list.append((field.name, field_dtype))
        else:
            dtype_list.append((field.name, field_dtype, (field.count,)))
        offset += base_dtype.itemsize * field.count

    while offset < msg.point_step:
        dtype_list.append((f"_pad{offset}", byte_order + "u1"))
        offset += 1

    dtype = np.dtype(dtype_list)
    points_data = np.frombuffer(msg.data, dtype=dtype)

    x = points_data["x"].astype(np.float32)
    y = points_data["y"].astype(np.float32)
    z = points_data["z"].astype(np.float32)
    t = points_data["t"].astype(np.float32)

    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    return np.stack((x[mask], y[mask], z[mask]), axis=-1), t[mask]


class KissSlamNode:
    def __init__(self):
        cloud_topic = rospy.get_param("~pointcloud_topic", "/points")
        config = rospy.get_param("~config", None)
        self.frame_id = rospy.get_param("~frame_id", "map")

        slam_config = load_config(config)
        self.kiss_slam = KissSLAM(slam_config)

        self.pose_pub = rospy.Publisher("~pose", PoseStamped, queue_size=1)
        self.sub = rospy.Subscriber(cloud_topic, PointCloud2, self.callback, queue_size=1)
        self.save_srv = rospy.Service("~save_map", Trigger, self.handle_save_map)

    def callback(self, msg: PointCloud2):
        points, timestamps = pointcloud2_to_numpy(msg)

        self.kiss_slam.process_scan(points, timestamps)

        keypose = self.kiss_slam.local_map_graph.last_keypose
        T_global = keypose @ self.kiss_slam.odometry.last_pose
        self.publish_pose(T_global, msg.header.stamp)

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
        output = rospy.get_param("~map_output", "/tmp/s1_1_kiss.pcd")
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

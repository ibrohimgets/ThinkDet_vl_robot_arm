"""
ROS2 node that wraps ThinkDet or GroundingDINO for language-conditioned picks.

The node stores the latest synchronized RGB-D frame, waits for a natural
language query on a topic, runs open-vocabulary detection, and publishes:
  - ranked `vision_msgs/Detection2DArray`
  - a 3D target point in the camera frame
  - a debug image with overlays

The runtime is parameterized so the same node can be used against a real RGB-D
camera or a simulator by swapping topic names and model paths.
"""

import threading

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from vision_msgs.msg import BoundingBox2D, Detection2D, Detection2DArray, ObjectHypothesisWithPose


class ThinkDetGraspNode(Node):
    """ROS wrapper for open-vocabulary detection plus RGB-D backprojection."""

    def __init__(self):
        super().__init__("thinkdet_grasp_node")

        self._declare_parameters()

        self.conf_thresh = float(self.get_parameter("confidence_threshold").value)
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.max_depth = float(self.get_parameter("max_depth_m").value)
        self.cam_frame = str(self.get_parameter("camera_frame").value).strip()
        self.top_k = int(self.get_parameter("top_k").value)

        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.query_topic = str(self.get_parameter("query_topic").value)
        self.detections_topic = str(self.get_parameter("detections_topic").value)
        self.grasp_target_topic = str(self.get_parameter("grasp_target_topic").value)
        self.debug_image_topic = str(self.get_parameter("debug_image_topic").value)
        self.sync_queue_size = int(self.get_parameter("sync_queue_size").value)
        self.sync_slop = float(self.get_parameter("sync_slop").value)

        extract_layer = int(self.get_parameter("extract_layer").value)
        extract_layers = self._parse_int_list_param(self.get_parameter("extract_layers").value)
        if extract_layer < 0:
            extract_layer = None

        model_path = str(self.get_parameter("model_path").value).strip()
        thinkdet_checkpoint = str(self.get_parameter("thinkdet_checkpoint").value).strip() or model_path

        self.get_logger().info("Loading inference runtime...")
        try:
            from ROS.thinkdet_runtime import ThinkDetInference
        except ImportError:
            from thinkdet_runtime import ThinkDetInference

        self.model = ThinkDetInference(
            repo_root=str(self.get_parameter("repo_root").value).strip(),
            backend=str(self.get_parameter("backend").value).strip(),
            device=str(self.get_parameter("device").value).strip(),
            gd_config=str(self.get_parameter("gd_config").value).strip(),
            gd_weights=str(self.get_parameter("gd_weights").value).strip(),
            internvl_path=str(self.get_parameter("internvl_path").value).strip(),
            thinkdet_checkpoint=thinkdet_checkpoint,
            extract_layer=extract_layer,
            extract_layers=extract_layers or None,
            layer_fusion=str(self.get_parameter("layer_fusion").value).strip() or None,
        )
        self.get_logger().info(f"Runtime ready: {self.model.describe()}")

        self.bridge = CvBridge()
        self.camera_K = None
        self._inference_lock = threading.Lock()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        rgb_sub = message_filters.Subscriber(
            self,
            Image,
            self.rgb_topic,
            qos_profile=sensor_qos,
        )
        depth_sub = message_filters.Subscriber(
            self,
            Image,
            self.depth_topic,
            qos_profile=sensor_qos,
        )
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
        )
        self.ts.registerCallback(self._rgbd_callback)

        self.cam_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._cam_info_callback,
            1,
        )

        self.query_sub = self.create_subscription(
            String,
            self.query_topic,
            self._query_callback,
            10,
        )

        self.det_pub = self.create_publisher(Detection2DArray, self.detections_topic, 10)
        self.grasp_pub = self.create_publisher(PointStamped, self.grasp_target_topic, 10)
        self.dbg_pub = self.create_publisher(Image, self.debug_image_topic, 1)

        self._latest_rgb = None
        self._latest_depth = None
        self._latest_stamp = None

        self.get_logger().info(
            "ThinkDetGraspNode initialized. Publish a query to "
            f"{self.query_topic} to run detection."
        )

    def _declare_parameters(self):
        self.declare_parameter("repo_root", "")
        self.declare_parameter("backend", "auto")
        self.declare_parameter("device", "auto")
        self.declare_parameter("model_path", "")
        self.declare_parameter("thinkdet_checkpoint", "")
        self.declare_parameter("gd_config", "")
        self.declare_parameter("gd_weights", "")
        self.declare_parameter("internvl_path", "")
        self.declare_parameter("extract_layer", -1)
        self.declare_parameter("extract_layers", "")
        self.declare_parameter("layer_fusion", "")
        self.declare_parameter("top_k", 5)

        self.declare_parameter("confidence_threshold", 0.30)
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("max_depth_m", 2.0)
        self.declare_parameter("camera_frame", "")

        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_rect_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("query_topic", "/thinkdet/query")
        self.declare_parameter("detections_topic", "/thinkdet/detections")
        self.declare_parameter("grasp_target_topic", "/thinkdet/grasp_target")
        self.declare_parameter("debug_image_topic", "/thinkdet/debug_image")
        self.declare_parameter("sync_queue_size", 5)
        self.declare_parameter("sync_slop", 0.05)

    def _parse_int_list_param(self, raw_value: str):
        text = str(raw_value or "").replace(",", " ")
        return [int(piece) for piece in text.split() if piece.strip()]

    def _cam_info_callback(self, msg: CameraInfo):
        if self.camera_K is None:
            self.camera_K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(f"Camera intrinsics received: fx={self.camera_K[0, 0]:.1f}")
            self.destroy_subscription(self.cam_info_sub)

    def _rgbd_callback(self, rgb_msg: Image, depth_msg: Image):
        self._latest_rgb = rgb_msg
        self._latest_depth = depth_msg
        self._latest_stamp = rgb_msg.header.stamp

    def _query_callback(self, msg: String):
        query = msg.data.strip()
        if not query:
            return

        if self._latest_rgb is None or self._latest_depth is None:
            self.get_logger().warn("No RGB-D frame received yet. Waiting...")
            return
        if self.camera_K is None:
            self.get_logger().warn("Camera intrinsics not yet received.")
            return
        if not self._inference_lock.acquire(blocking=False):
            self.get_logger().warn("Inference already running; dropping new query.")
            return

        self.get_logger().info(f"Query received: '{query}'")
        rgb_msg = self._latest_rgb
        depth_msg = self._latest_depth
        stamp = self._latest_stamp
        thread = threading.Thread(
            target=self._run_query_job,
            args=(query, rgb_msg, depth_msg, stamp),
            daemon=True,
        )
        thread.start()

    def _run_query_job(self, query: str, rgb_msg: Image, depth_msg: Image, stamp):
        try:
            self._run_inference(query, rgb_msg, depth_msg, stamp)
        except Exception as exc:
            self.get_logger().error(f"Inference failed for query '{query}': {exc}")
        finally:
            self._inference_lock.release()

    def _run_inference(self, query: str, rgb_msg: Image, depth_msg: Image, stamp):
        rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        results = self.model.predict(
            image=rgb_cv,
            query=query,
            top_k=self.top_k,
            score_threshold=self.conf_thresh,
        )
        if not results:
            self.get_logger().warn(f"No detections above threshold for query: '{query}'")
            return

        results.sort(key=lambda r: r.get("score", 0.0), reverse=True)

        frame_id = self._resolve_frame_id(rgb_msg, depth_msg)
        det_array = self._make_detection2d_array(results, stamp, frame_id)
        self.det_pub.publish(det_array)

        top = results[0]
        x1, y1, x2, y2 = [int(v) for v in top["bbox"]]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        grasp_3d = self._backproject(cx, cy, depth_cv, bbox=top["bbox"])
        if grasp_3d is not None:
            pt_msg = PointStamped()
            pt_msg.header.stamp = stamp
            pt_msg.header.frame_id = frame_id
            pt_msg.point.x, pt_msg.point.y, pt_msg.point.z = grasp_3d
            self.grasp_pub.publish(pt_msg)
            self.get_logger().info(
                f"Grasp target: ({grasp_3d[0]:.3f}, {grasp_3d[1]:.3f}, {grasp_3d[2]:.3f}) m "
                f"| score={top['score']:.2f} | label={top.get('label', '?')}"
            )
        else:
            self.get_logger().warn("Depth value invalid inside the selected detection.")

        debug_img = self._draw_detections(rgb_cv.copy(), results, query)
        dbg_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
        dbg_msg.header = rgb_msg.header
        self.dbg_pub.publish(dbg_msg)

    def _resolve_frame_id(self, rgb_msg: Image, depth_msg: Image) -> str:
        if self.cam_frame:
            return self.cam_frame
        if rgb_msg.header.frame_id:
            return rgb_msg.header.frame_id
        if depth_msg.header.frame_id:
            return depth_msg.header.frame_id
        return "camera_color_optical_frame"

    def _project_pixel(self, px: int, py: int, depth_m: float):
        fx, fy = self.camera_K[0, 0], self.camera_K[1, 1]
        ppx, ppy = self.camera_K[0, 2], self.camera_K[1, 2]
        x = (px - ppx) * depth_m / fx
        y = (py - ppy) * depth_m / fy
        return (x, y, depth_m)

    def _backproject(self, cx: int, cy: int, depth: np.ndarray, bbox=None):
        h, w = depth.shape[:2]
        r = 3
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        patch = depth[y0:y1, x0:x1].astype(np.float32)
        if depth.dtype == np.uint16:
            patch = patch * self.depth_scale
        valid = patch[(patch > 0.05) & (patch < self.max_depth)]
        if valid.size != 0:
            return self._project_pixel(cx, cy, float(np.median(valid)))

        if bbox is None:
            return None

        bx1, by1, bx2, by2 = [int(v) for v in bbox]
        bx1 = max(0, min(w - 1, bx1))
        bx2 = max(bx1 + 1, min(w, bx2))
        by1 = max(0, min(h - 1, by1))
        by2 = max(by1 + 1, min(h, by2))
        crop = depth[by1:by2, bx1:bx2].astype(np.float32)
        if depth.dtype == np.uint16:
            crop = crop * self.depth_scale
        valid_mask = (crop > 0.05) & (crop < self.max_depth)
        if not np.any(valid_mask):
            return None

        valid_pixels = np.argwhere(valid_mask)
        target_xy = np.array([cy - by1, cx - bx1], dtype=np.float32)
        offsets = valid_pixels.astype(np.float32) - target_xy
        best_idx = int(np.argmin(np.sum(offsets * offsets, axis=1)))
        py, px = valid_pixels[best_idx]
        depth_m = float(crop[py, px])
        return self._project_pixel(bx1 + int(px), by1 + int(py), depth_m)

    def _set_bbox_center(self, bbox_msg: BoundingBox2D, x: float, y: float):
        if hasattr(bbox_msg.center, "position"):
            bbox_msg.center.position.x = float(x)
            bbox_msg.center.position.y = float(y)
        else:
            bbox_msg.center.x = float(x)
            bbox_msg.center.y = float(y)
            if hasattr(bbox_msg.center, "theta"):
                bbox_msg.center.theta = 0.0

    def _make_detection2d_array(self, results, stamp, frame_id: str) -> Detection2DArray:
        arr = Detection2DArray()
        arr.header.stamp = stamp
        arr.header.frame_id = frame_id
        for r in results:
            d = Detection2D()
            if hasattr(d, "header"):
                d.header.stamp = stamp
                d.header.frame_id = frame_id
            x1, y1, x2, y2 = r["bbox"]
            bb = BoundingBox2D()
            self._set_bbox_center(bb, (x1 + x2) / 2, (y1 + y2) / 2)
            bb.size_x = float(x2 - x1)
            bb.size_y = float(y2 - y1)
            d.bbox = bb
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.score = float(r.get("score", 1.0))
            hyp.hypothesis.class_id = str(r.get("label", "object"))
            d.results.append(hyp)
            arr.detections.append(d)
        return arr

    def _draw_detections(self, img: np.ndarray, results, query: str) -> np.ndarray:
        colors = [(0, 255, 0), (0, 200, 255), (255, 100, 0), (200, 0, 200)]
        for i, r in enumerate(results):
            x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
            color = colors[i % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"#{i + 1} {r.get('label', '?')} {r.get('score', 0):.2f}"
            cv2.putText(
                img,
                label,
                (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            img,
            f"Q: {query}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return img


def main(args=None):
    rclpy.init(args=args)
    node = ThinkDetGraspNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

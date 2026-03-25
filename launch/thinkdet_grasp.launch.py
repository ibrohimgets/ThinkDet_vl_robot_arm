import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    repo_root_default = os.environ.get("THINKDET_REPO_ROOT", "")

    args = [
        DeclareLaunchArgument("repo_root", default_value=repo_root_default),
        DeclareLaunchArgument("backend", default_value="auto"),
        DeclareLaunchArgument("device", default_value="auto"),
        DeclareLaunchArgument("thinkdet_checkpoint", default_value=""),
        DeclareLaunchArgument("gd_config", default_value=""),
        DeclareLaunchArgument("gd_weights", default_value=""),
        DeclareLaunchArgument("internvl_path", default_value=""),
        DeclareLaunchArgument("rgb_topic", default_value="/camera/color/image_raw"),
        DeclareLaunchArgument("depth_topic", default_value="/camera/depth/image_rect_raw"),
        DeclareLaunchArgument("camera_info_topic", default_value="/camera/color/camera_info"),
        DeclareLaunchArgument("query_topic", default_value="/thinkdet/query"),
        DeclareLaunchArgument("camera_frame", default_value=""),
        DeclareLaunchArgument("confidence_threshold", default_value="0.30"),
        DeclareLaunchArgument("top_k", default_value="5"),
    ]

    node = Node(
        package="thinkdet_ros2_demo",
        executable="thinkdet_grasp_node",
        name="thinkdet_grasp_node",
        output="screen",
        parameters=[
            {
                "repo_root": LaunchConfiguration("repo_root"),
                "backend": LaunchConfiguration("backend"),
                "device": LaunchConfiguration("device"),
                "thinkdet_checkpoint": LaunchConfiguration("thinkdet_checkpoint"),
                "gd_config": LaunchConfiguration("gd_config"),
                "gd_weights": LaunchConfiguration("gd_weights"),
                "internvl_path": LaunchConfiguration("internvl_path"),
                "rgb_topic": LaunchConfiguration("rgb_topic"),
                "depth_topic": LaunchConfiguration("depth_topic"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                "query_topic": LaunchConfiguration("query_topic"),
                "camera_frame": LaunchConfiguration("camera_frame"),
                "confidence_threshold": LaunchConfiguration("confidence_threshold"),
                "top_k": LaunchConfiguration("top_k"),
            }
        ],
    )

    return LaunchDescription(args + [node])

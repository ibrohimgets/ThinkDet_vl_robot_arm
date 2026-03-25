from glob import glob
from setuptools import find_packages, setup


package_name = "thinkdet_ros2_demo"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=["ROS", "ROS.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="iibrohimm",
    maintainer_email="iibrohimm@local",
    description="ROS2 package for ThinkDet and GroundingDINO robot demos.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "thinkdet_grasp_node = ROS.thinkdet_ros2_node:main",
        ],
    },
)

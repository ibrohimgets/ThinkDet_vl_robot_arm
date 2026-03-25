from glob import glob
from setuptools import setup


package_name = "thinkdet_ros2_demo"


setup(
    name=package_name,
    version="0.1.0",
    py_modules=["thinkdet_ros2_node", "thinkdet_runtime"],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "README.md", "LICENSE"]),
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
            "thinkdet_grasp_node = thinkdet_ros2_node:main",
        ],
    },
)

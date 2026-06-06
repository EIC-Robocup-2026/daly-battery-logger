"""
Optional ROS2 publishing bridge.

rclpy is NOT pip-installable — install via system packages or rosdep before use:
  sudo apt install ros-<distro>-rclpy ros-<distro>-sensor-msgs

When rclpy is not present the module still imports cleanly and
BMSRos2Publisher.enabled is always False.
"""

try:
    import rclpy  # noqa: F401
    from rclpy.node import Node
    from sensor_msgs.msg import BatteryState

    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    Node = object  # type: ignore[assignment,misc]


class BMSRos2Publisher(Node):  # type: ignore[misc]
    def __init__(self, node_name: str):
        if not HAS_ROS2:
            self.enabled = False
            return
        super().__init__(node_name)
        self.enabled = True
        topic = f"{node_name}/battery_state"
        self._pub = self.create_publisher(BatteryState, topic, 10)

    def publish(self, soc_data: dict, temp_data: dict | None = None):
        if not self.enabled:
            return
        msg = BatteryState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.voltage = float(soc_data.get("total_voltage") or float("nan"))
        msg.current = float(soc_data.get("current") or float("nan"))
        pct = soc_data.get("soc_percent")
        msg.percentage = float(pct) / 100.0 if pct is not None else float("nan")
        if temp_data:
            t = temp_data.get("highest_temperature")
            msg.temperature = float(t) if t is not None else float("nan")
        msg.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_UNKNOWN
        self._pub.publish(msg)

#!/usr/bin/env python3

import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped

from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool
from mavros_msgs.srv import SetMode


class DronePurePursuit:

    def __init__(self):

        rospy.init_node("drone_pure_pursuit")

        # ==========================
        # Parameters
        # ==========================

        self.max_speed = rospy.get_param("~max_speed", 2.0)

        self.lookahead_min = rospy.get_param(
            "~lookahead_min",
            1.5
        )

        self.lookahead_max = rospy.get_param(
            "~lookahead_max",
            5.0
        )

        self.goal_tolerance = rospy.get_param(
            "~goal_tolerance",
            0.5
        )

        # ==========================
        # Vehicle State
        # ==========================

        self.current_state = State()

        self.current_position = np.zeros(3)

        self.pose_received = False

        # ==========================
        # Path
        # ==========================

        self.path = self.load_path()

        self.current_wp_idx = 0

        # ==========================
        # Subscribers
        # ==========================

        rospy.Subscriber(
            "/mavros/state",
            State,
            self.state_cb
        )

        rospy.Subscriber(
            "/mavros/local_position/pose",
            PoseStamped,
            self.pose_cb
        )

        # ==========================
        # Publishers
        # ==========================

        self.vel_pub = rospy.Publisher(
            "/mavros/setpoint_velocity/cmd_vel",
            TwistStamped,
            queue_size=20
        )

        # ==========================
        # Services
        # ==========================

        rospy.wait_for_service("/mavros/cmd/arming")
        rospy.wait_for_service("/mavros/set_mode")

        self.arm_client = rospy.ServiceProxy(
            "/mavros/cmd/arming",
            CommandBool
        )

        self.mode_client = rospy.ServiceProxy(
            "/mavros/set_mode",
            SetMode
        )

        rospy.loginfo("Drone Pure Pursuit Ready")

    # =====================================================
    # Callbacks
    # =====================================================

    def state_cb(self, msg):
        self.current_state = msg

    def pose_cb(self, msg):

        self.current_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        self.pose_received = True

    # =====================================================
    # Path
    # =====================================================

    def load_path(self):

        path = [

            [0, 0, 3],

            [10, 0, 3],

            [10, 10, 3],

            [0, 10, 3],

            [0, 0, 3]

        ]

        return np.array(path)

    # =====================================================
    # Dynamic Lookahead
    # =====================================================

    def compute_lookahead(self):

        speed_ratio = min(
            self.max_speed / 5.0,
            1.0
        )

        return (
            self.lookahead_min +
            speed_ratio *
            (self.lookahead_max -
             self.lookahead_min)
        )

    # =====================================================
    # Pure Pursuit
    # =====================================================

    def find_target_point(self):

        lookahead = self.compute_lookahead()

        for i in range(
            self.current_wp_idx,
            len(self.path)
        ):

            wp = self.path[i]

            dist = np.linalg.norm(
                wp - self.current_position
            )

            if dist >= lookahead:

                self.current_wp_idx = i

                return wp

        return self.path[-1]

    def compute_velocity_command(self):

        target = self.find_target_point()

        vec = target - self.current_position

        dist = np.linalg.norm(vec)

        if dist < 0.001:
            return np.zeros(3)

        direction = vec / dist

        speed = min(
            self.max_speed,
            dist
        )

        return direction * speed

    # =====================================================
    # Goal Check
    # =====================================================

    def reached_goal(self):

        goal = self.path[-1]

        error = np.linalg.norm(
            goal - self.current_position
        )

        return error < self.goal_tolerance

    # =====================================================
    # Publish Velocity
    # =====================================================

    def publish_velocity(self, vel):

        msg = TwistStamped()

        msg.header.stamp = rospy.Time.now()

        msg.twist.linear.x = vel[0]
        msg.twist.linear.y = vel[1]
        msg.twist.linear.z = vel[2]

        self.vel_pub.publish(msg)

    # =====================================================
    # Offboard Setup
    # =====================================================

    def start_offboard(self):

        rospy.loginfo(
            "Sending initial setpoints..."
        )

        rate = rospy.Rate(20)

        zero = np.zeros(3)

        for _ in range(100):

            self.publish_velocity(zero)

            rate.sleep()

        rospy.loginfo("Switching OFFBOARD")

        self.mode_client(
            custom_mode="OFFBOARD"
        )

        rospy.sleep(1.0)

        rospy.loginfo("Arming")

        self.arm_client(True)

    # =====================================================
    # Main Loop
    # =====================================================

    def run(self):

        rate = rospy.Rate(20)

        while (
            not rospy.is_shutdown()
            and not self.current_state.connected
        ):
            rate.sleep()

        rospy.loginfo(
            "Connected to FCU"
        )

        while (
            not rospy.is_shutdown()
            and not self.pose_received
        ):
            rate.sleep()

        self.start_offboard()

        while not rospy.is_shutdown():

            if self.reached_goal():

                rospy.loginfo(
                    "Mission Complete"
                )

                self.publish_velocity(
                    np.zeros(3)
                )

                break

            vel = self.compute_velocity_command()

            self.publish_velocity(vel)

            rate.sleep()


if __name__ == "__main__":

    node = DronePurePursuit()

    node.run()

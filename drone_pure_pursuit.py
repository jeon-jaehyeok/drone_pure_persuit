#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pure Pursuit target selector for PX4 drone + DWA mission.

This node does NOT publish a velocity command directly.
It publishes only the lookahead target point for the DWA planner.

Inputs:
  /mavros/local_position/pose     geometry_msgs/PoseStamped
  /mavros/state                   mavros_msgs/State

Outputs:
  /pure_pursuit/local_goal        geometry_msgs/PointStamped   ← DWA가 구독
  /pure_pursuit/final_reached     std_msgs/Bool
  /pure_pursuit/path              nav_msgs/Path                ← RViz 디버그용
  /pure_pursuit/lookahead_dist    std_msgs/Float32             ← 디버그용

Pure pursuit details:
  - CSV 파일(x, y, z 컬럼)에서 경로를 로드합니다.
  - 적응형 lookahead: Ld = clamp(gain * speed_xy, min, max)
  - 경로 진행이 단조 증가(backward jump 방지)
  - lookahead circle과 경로 세그먼트의 교점을 목표점으로 선택
  - 교점 없을 경우 경로 호 길이를 따라 fallback
"""

import csv
import math
import os

import numpy as np
import rospy

from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool, Float32

from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode


class DronePurePursuit:

    def __init__(self):

        rospy.init_node("drone_pure_pursuit")

        # ==========================
        # Parameters
        # ==========================

        self.csv_path = rospy.get_param(
            "~csv_path", ""
        )

        self.frame_id = rospy.get_param(
            "~frame_id", "map"
        )

        self.default_altitude = float(rospy.get_param(
            "~default_altitude", 3.0
        ))

        # True: CSV의 z값 무시하고 default_altitude 고정 사용
        self.force_path_altitude = bool(rospy.get_param(
            "~force_path_altitude", False
        ))

        # --- Lookahead ---
        self.lookahead_distance = float(rospy.get_param(
            "~lookahead_distance", 1.5
        ))

        self.use_adaptive_lookahead = bool(rospy.get_param(
            "~use_adaptive_lookahead", True
        ))

        self.lookahead_gain = float(rospy.get_param(
            "~lookahead_gain", 1.0
        ))

        self.lookahead_min = float(rospy.get_param(
            "~lookahead_min", 1.5
        ))

        self.lookahead_max = float(rospy.get_param(
            "~lookahead_max", 5.0
        ))

        # --- Goal tolerance ---
        self.goal_tolerance_xy = float(rospy.get_param(
            "~goal_tolerance_xy", 0.5
        ))

        self.goal_tolerance_z = float(rospy.get_param(
            "~goal_tolerance_z", 0.8
        ))

        # --- Topics ---
        self.goal_topic = rospy.get_param(
            "~goal_topic", "/pure_pursuit/local_goal"
        )

        self.final_reached_topic = rospy.get_param(
            "~final_reached_topic", "/pure_pursuit/final_reached"
        )

        self.publish_rate = float(rospy.get_param(
            "~publish_rate", 20.0
        ))

        self.pose_timeout = float(rospy.get_param(
            "~pose_timeout", 0.5
        ))

        # --- Path search ---
        self.nearest_search_back_segments = int(rospy.get_param(
            "~nearest_search_back_segments", 5
        ))

        # ==========================
        # Path
        # ==========================

        self.points = self.load_csv(self.csv_path)

        if not self.points:
            rospy.logwarn(
                "csv_path가 비어 있거나 경로를 찾을 수 없습니다. "
                "하드코딩된 기본 경로를 사용합니다: %s",
                self.csv_path
            )
            self.points = self._default_path()

        self.segment_lengths = self._compute_segment_lengths(self.points)

        # ==========================
        # Vehicle State
        # ==========================

        self.current_state = State()
        self.position = [0.0, 0.0, 0.0]
        self.speed_xy = 0.0

        self.has_pose = False
        self.last_pose_receive_time = rospy.Time(0)
        self.last_position = None
        self.last_pose_time = None

        # Pure pursuit 진행 상태
        self.progress_s = 0.0          # 경로 진행 위치 (segment_index + t)
        self.last_target_index = 0
        self.final_reached = False

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

        self.goal_pub = rospy.Publisher(
            self.goal_topic,
            PointStamped,
            queue_size=10
        )

        self.final_pub = rospy.Publisher(
            self.final_reached_topic,
            Bool,
            queue_size=1,
            latch=True
        )

        self.path_pub = rospy.Publisher(
            "/pure_pursuit/path",
            Path,
            queue_size=1,
            latch=True
        )

        self.lookahead_pub = rospy.Publisher(
            "/pure_pursuit/lookahead_dist",
            Float32,
            queue_size=10
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

        rospy.loginfo(
            "DronePurePursuit 초기화 완료 | 경로 포인트 수: %d | "
            "adaptive lookahead: %s (min=%.1f, max=%.1f)",
            len(self.points),
            self.use_adaptive_lookahead,
            self.lookahead_min,
            self.lookahead_max
        )

    # =====================================================
    # Callbacks
    # =====================================================

    def state_cb(self, msg):
        self.current_state = msg

    def pose_cb(self, msg):
        p = msg.pose.position
        now = msg.header.stamp

        if now == rospy.Time(0):
            now = rospy.Time.now()

        # 속도 추정 (이전 포즈와의 차이)
        if self.last_position is not None and self.last_pose_time is not None:
            dt = (now - self.last_pose_time).to_sec()
            if dt > 1e-4:
                dx = p.x - self.last_position[0]
                dy = p.y - self.last_position[1]
                self.speed_xy = math.hypot(dx, dy) / dt

        self.position = [p.x, p.y, p.z]
        self.last_position = [p.x, p.y, p.z]
        self.last_pose_time = now
        self.last_pose_receive_time = rospy.Time.now()
        self.has_pose = True

    # =====================================================
    # CSV 로드
    # =====================================================

    def load_csv(self, path):
        if not path:
            return []
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return []

        points = []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "x" in reader.fieldnames and "y" in reader.fieldnames:
                # x, y, z 헤더가 있는 경우
                for row in reader:
                    z = float(row.get("z", self.default_altitude) or self.default_altitude)
                    if self.force_path_altitude:
                        z = self.default_altitude
                    points.append([float(row["x"]), float(row["y"]), z])
            else:
                # 헤더 없이 raw x,y,z 값만 있는 경우
                f.seek(0)
                raw = csv.reader(f)
                for row in raw:
                    if len(row) < 2:
                        continue
                    try:
                        x = float(row[0])
                        y = float(row[1])
                        z = float(row[2]) if len(row) >= 3 else self.default_altitude
                        if self.force_path_altitude:
                            z = self.default_altitude
                        points.append([x, y, z])
                    except ValueError:
                        continue
        return points

    def _default_path(self):
        """csv_path 미설정 시 사용하는 기본 사각형 경로"""
        z = self.default_altitude
        return [
            [0.0,  0.0,  z],
            [10.0, 0.0,  z],
            [10.0, 10.0, z],
            [0.0,  10.0, z],
            [0.0,  0.0,  z],
        ]

    # =====================================================
    # 기하 헬퍼
    # =====================================================

    @staticmethod
    def _compute_segment_lengths(points):
        lengths = []
        for i in range(len(points) - 1):
            lengths.append(math.hypot(
                points[i + 1][0] - points[i][0],
                points[i + 1][1] - points[i][1]
            ))
        return lengths

    @staticmethod
    def _dist_xy(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _clamp(x, lo, hi):
        return max(lo, min(hi, x))

    @staticmethod
    def _interpolate(a, b, t):
        return [
            a[0] + t * (b[0] - a[0]),
            a[1] + t * (b[1] - a[1]),
            a[2] + t * (b[2] - a[2]),
        ]

    @staticmethod
    def _project_point_to_segment(p, a, b):
        """점 p를 세그먼트 a-b에 투영. (t, 거리) 반환"""
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        denom = dx * dx + dy * dy
        if denom < 1e-12:
            return 0.0, math.hypot(p[0] - a[0], p[1] - a[1])
        t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / denom
        t = max(0.0, min(1.0, t))
        qx = a[0] + t * dx
        qy = a[1] + t * dy
        return t, math.hypot(p[0] - qx, p[1] - qy)

    def _segment_circle_intersections(self, a, b, center, radius):
        """세그먼트 a-b와 lookahead circle의 교점 t 값 목록 반환"""
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        fx = a[0] - center[0]
        fy = a[1] - center[1]

        A = dx * dx + dy * dy
        if A < 1e-12:
            return []
        B = 2.0 * (fx * dx + fy * dy)
        C = fx * fx + fy * fy - radius * radius
        disc = B * B - 4.0 * A * C
        if disc < 0.0:
            return []
        disc = math.sqrt(max(0.0, disc))
        t1 = (-B - disc) / (2.0 * A)
        t2 = (-B + disc) / (2.0 * A)

        ts = []
        for t in (t1, t2):
            if -1e-9 <= t <= 1.0 + 1e-9:
                ts.append(max(0.0, min(1.0, t)))
        return sorted(set(round(t, 10) for t in ts))

    # =====================================================
    # Adaptive Lookahead
    # =====================================================

    def get_lookahead_distance(self):
        if not self.use_adaptive_lookahead:
            return self.lookahead_distance
        adaptive = self.lookahead_gain * max(0.0, self.speed_xy)
        return self._clamp(adaptive, self.lookahead_min, self.lookahead_max)

    # =====================================================
    # Pure Pursuit Target Selection
    # =====================================================

    def _find_closest_segment(self):
        """현재 위치에서 가장 가까운 경로 세그먼트 탐색 (단조 진행 보장)"""
        if len(self.points) == 1:
            return 0, 0.0, 0.0

        start = max(0, int(math.floor(self.progress_s)) - self.nearest_search_back_segments)
        best_i, best_t, best_d = start, 0.0, float("inf")

        for i in range(start, len(self.points) - 1):
            t, d = self._project_point_to_segment(
                self.position, self.points[i], self.points[i + 1]
            )
            s = i + t
            # backward jump 방지: 현재 진행보다 0.5 이상 뒤로 가지 않음
            if s + 1e-6 < self.progress_s - 0.5:
                continue
            if d < best_d:
                best_i, best_t, best_d = i, t, d

        # Recovery: 장애물 회피 후 경로에서 멀어진 경우 전체 경로 재탐색
        if best_d == float("inf"):
            for i in range(len(self.points) - 1):
                t, d = self._project_point_to_segment(
                    self.position, self.points[i], self.points[i + 1]
                )
                if d < best_d:
                    best_i, best_t, best_d = i, t, d

        return best_i, best_t, best_d

    def _point_along_path(self, seg_i, seg_t, distance_ahead):
        """경로 호 길이를 따라 distance_ahead만큼 전진한 점 반환 (fallback)"""
        if len(self.points) == 1:
            return self.points[0], 0

        remain = max(0.0, distance_ahead)
        i = min(max(0, seg_i), len(self.points) - 2)
        t = self._clamp(seg_t, 0.0, 1.0)

        while i < len(self.points) - 1:
            seg_len = self.segment_lengths[i]
            if seg_len < 1e-12:
                i += 1
                continue
            available = seg_len * (1.0 - t) if i == seg_i else seg_len
            if remain <= available + 1e-9:
                start_t = t if i == seg_i else 0.0
                new_t = start_t + remain / seg_len
                target = self._interpolate(
                    self.points[i], self.points[i + 1],
                    self._clamp(new_t, 0.0, 1.0)
                )
                return target, i + 1
            remain -= available
            i += 1
            t = 0.0

        return self.points[-1], len(self.points) - 1

    def select_target(self):
        """lookahead distance에 따라 목표 경로점 선택"""
        if len(self.points) == 1:
            self.progress_s = 0.0
            self.last_target_index = 0
            return self.points[0], 0

        lookahead = self.get_lookahead_distance()
        seg_i, seg_t, _ = self._find_closest_segment()

        closest_s = seg_i + seg_t
        self.progress_s = max(self.progress_s, closest_s)

        # 1순위: lookahead circle과 경로 세그먼트의 교점 탐색
        start_seg = max(0, min(int(math.floor(self.progress_s)), len(self.points) - 2))
        for i in range(start_seg, len(self.points) - 1):
            ts = self._segment_circle_intersections(
                self.points[i], self.points[i + 1],
                self.position, lookahead
            )
            valid = [t for t in ts if (i + t) + 1e-6 >= self.progress_s]
            if valid:
                t = max(valid)  # circle을 빠져나가는 전방 교점 선택
                target = self._interpolate(self.points[i], self.points[i + 1], t)
                self.last_target_index = min(len(self.points) - 1, i + 1)
                return target, self.last_target_index

        # 2순위: 호 길이 fallback (경로 시작 지점에서 멀리 있을 때)
        target, idx = self._point_along_path(seg_i, seg_t, lookahead)
        self.last_target_index = max(self.last_target_index, idx)
        return target, self.last_target_index

    # =====================================================
    # Goal Check
    # =====================================================

    def check_final_reached(self):
        final = self.points[-1]
        xy_ok = self._dist_xy(self.position, final) <= self.goal_tolerance_xy
        z_ok = (abs(self.position[2] - final[2]) <= self.goal_tolerance_z
                or final[2] <= 0.2)  # 착륙 포인트는 z 무시
        near_end = self.last_target_index >= len(self.points) - 1
        return near_end and xy_ok and z_ok

    # =====================================================
    # Publishers
    # =====================================================

    def publish_goal(self, point, idx):
        msg = PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.frame_id
        msg.point.x = point[0]
        msg.point.y = point[1]
        msg.point.z = point[2]
        self.goal_pub.publish(msg)

        self.lookahead_pub.publish(
            Float32(data=self.get_lookahead_distance())
        )

        rospy.logdebug(
            "local_goal → (%.2f, %.2f, %.2f) | idx=%d | Ld=%.2f",
            point[0], point[1], point[2],
            idx, self.get_lookahead_distance()
        )

    def publish_path(self):
        """전체 경로를 RViz 확인용으로 한 번 발행"""
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = self.frame_id
        for p in self.points:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = p[0]
            ps.pose.position.y = p[1]
            ps.pose.position.z = p[2]
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
        self.path_pub.publish(path_msg)

    # =====================================================
    # Offboard Setup
    # =====================================================

    def start_offboard(self):
        rospy.loginfo("초기 setpoint 전송 중...")
        rate = rospy.Rate(20)

        # OFFBOARD 전환 전 setpoint를 미리 채워야 함
        dummy = PointStamped()
        dummy.header.frame_id = self.frame_id
        dummy.point.x = self.position[0]
        dummy.point.y = self.position[1]
        dummy.point.z = self.position[2]

        for _ in range(100):
            dummy.header.stamp = rospy.Time.now()
            self.goal_pub.publish(dummy)
            rate.sleep()

        rospy.loginfo("OFFBOARD 모드 전환")
        self.mode_client(custom_mode="OFFBOARD")
        rospy.sleep(1.0)

        rospy.loginfo("Arming")
        self.arm_client(True)

    # =====================================================
    # Main Loop
    # =====================================================

    def is_pose_fresh(self):
        if not self.has_pose:
            return False
        age = (rospy.Time.now() - self.last_pose_receive_time).to_sec()
        return age <= self.pose_timeout

    def run(self):
        rate = rospy.Rate(self.publish_rate)

        # FCU 연결 대기
        while not rospy.is_shutdown() and not self.current_state.connected:
            rate.sleep()
        rospy.loginfo("FCU 연결됨")

        # 첫 포즈 수신 대기
        while not rospy.is_shutdown() and not self.has_pose:
            rate.sleep()
        rospy.loginfo("포즈 수신 완료")

        # 경로 발행 (RViz 디버그용)
        self.publish_path()

        # OFFBOARD + Arming
        self.start_offboard()

        while not rospy.is_shutdown():

            if not self.is_pose_fresh():
                rospy.logwarn_throttle(
                    1.0, "포즈 타임아웃 — setpoint 발행 중단"
                )
                rate.sleep()
                continue

            # 최종 목표 도달 확인
            self.final_reached = self.check_final_reached()
            self.final_pub.publish(Bool(data=self.final_reached))

            if self.final_reached:
                rospy.loginfo("미션 완료 — 최종 목표 도달")
                # 마지막 포인트를 계속 발행 (DWA가 정지 유지)
                final_goal = PointStamped()
                final_goal.header.stamp = rospy.Time.now()
                final_goal.header.frame_id = self.frame_id
                final_goal.point.x = self.points[-1][0]
                final_goal.point.y = self.points[-1][1]
                final_goal.point.z = self.points[-1][2]
                self.goal_pub.publish(final_goal)
                break

            # lookahead로 목표 경로점 선택 후 발행
            target, idx = self.select_target()
            self.publish_goal(target, idx)

            rate.sleep()


if __name__ == "__main__":
    node = DronePurePursuit()
    node.run()

"""One-shot tabletop pick from the Panther RGB-D cameras and an existing VOP stream."""
from __future__ import annotations

import json
import multiprocessing
import os
import queue
import threading
import time
import zlib
from uuid import uuid4

import numpy as np

from common.vendor_runtime import tool
from contracts import HEAD_JOINTS, finite_number, finite_vector, require_confirmation
from panther_bridge import put_latest
from panther_motion import GRIPPER_MODES, notify_async
from panther_payloads import DEPTH_OUTPUT_SIZE

CARD = "tabletop_pick"
MAX_AGE_S = 1.0
MAX_SKEW_S = 0.15
VOP_LIMIT = 256 * 1024


def read_vop(profile, domain, topic, messages, stop):
    """Subscribe on the core RMW in a separate process, as in the outbound bridge."""
    if not os.path.isfile(profile):
        return
    os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = profile
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import String

    rclpy.init(domain_id=domain)
    node = Node("panther_tabletop_pick_vop")

    def receive(msg):
        if len(msg.data) <= VOP_LIMIT:
            put_latest(messages, msg.data)

    try:
        node.create_subscription(String, topic, receive, qos_profile_sensor_data)
        while rclpy.ok() and not stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class VopInput:
    def __init__(self, profile, domain):
        self._profile, self._domain = profile, domain
        self._ctx = multiprocessing.get_context("spawn")
        self._process = self._queue = self._stop = self._latest = None

    def start(self, topic):
        if self._process is not None and self._process.is_alive():
            return
        self.close()
        if not os.path.isfile(self._profile):
            raise RuntimeError("Panther core DDS profile is missing")
        self._queue, self._stop = self._ctx.Queue(maxsize=2), self._ctx.Event()
        self._process = self._ctx.Process(target=read_vop, daemon=True,
                                         args=(self._profile, self._domain, topic, self._queue, self._stop))
        try:
            self._process.start()
        except Exception:
            self._process = None
            self.close()
            raise

    def latest(self):
        if self._process is None or not self._process.is_alive():
            return None
        while True:
            try:
                self._latest = self._queue.get_nowait()
            except queue.Empty:
                break
        try:
            data = json.loads(self._latest)
        except (TypeError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def close(self):
        if self._process is not None:
            self._stop.set()
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
            self._process.close()
        if self._queue is not None:
            self._queue.close()
        self._process = self._queue = self._stop = self._latest = None


def load_setup(config):
    if config.get("depth_aligned_to_color") is not True:
        raise ValueError("configure an aligned RGB-D source and set depth_aligned_to_color")
    intrinsics = finite_vector(config.get("color_intrinsics"), "color_intrinsics", 4)
    size = finite_vector(config.get("color_size"), "color_size", 2)
    if min(intrinsics[:2]) <= 0 or min(size) < 2 or any(int(x) != x for x in size):
        raise ValueError("invalid rectified color intrinsics or image dimensions")
    transform = np.array(finite_vector(config.get("base_from_color"), "base_from_color", 16)).reshape(4, 4)
    rotation = transform[:3, :3]
    if (not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4)):
        raise ValueError("base_from_color must be a row-major rigid transform")
    bounds = np.array(finite_vector(config.get("workspace_m"), "workspace_m", 6)).reshape(2, 3)
    if np.any(bounds[0] >= bounds[1]):
        raise ValueError("workspace_m requires [min_x, min_y, min_z, max_x, max_y, max_z]")
    orientation = finite_vector(config.get("tool_orientation_xyzw"), "tool_orientation_xyzw", 4)
    if not np.isclose(np.linalg.norm(orientation), 1, atol=1e-4):
        raise ValueError("tool_orientation_xyzw must be a unit quaternion")
    head = finite_vector(config.get("head_rad"), "head_rad", 2)
    lift = finite_number(config.get("lift_position"), "lift_position")
    clearance = finite_number(config.get("clearance_m"), "clearance_m")
    if not 0.02 <= clearance <= 0.25:
        raise ValueError("clearance_m must be within [0.02, 0.25]")
    side = config.get("side")
    group, link = config.get("moveit_group"), config.get("tool_link")
    if side not in ("left", "right") or not all(isinstance(s, str) and s.strip() for s in (group, link)):
        raise ValueError("configure side, moveit_group and tool_link")
    return intrinsics, size, transform, bounds, orientation, head, lift, clearance


def select_object(data, name, color, confidence):
    objects = data.get("objects")
    if not isinstance(objects, list) or len(objects) > 1000:
        raise ValueError("VOP objects payload is invalid")
    matches = []
    for obj in objects:
        if not isinstance(obj, dict) or str(obj.get("name", "")).strip().casefold() != name.casefold():
            continue
        label = obj.get("color")
        if isinstance(label, dict):
            label = label.get("color_name") or label.get("dominant_hue")
        if color and color.casefold() not in str(label or "").casefold().split():
            continue
        if finite_number(obj.get("confidence"), "confidence") >= confidence:
            matches.append(obj)
    if len(matches) != 1:
        raise ValueError(f"expected one matching VOP detection, found {len(matches)}")
    return matches[0]


def grasp_points(box, payload, setup, surface_offset):
    intrinsics, size, transform, bounds, _, _, _, clearance = setup
    width, height = size
    x1, y1, x2, y2 = finite_vector(box, "VOP bbox", 4)
    if not (0 < x1 < x2 < width and 0 < y1 < y2 < height):
        raise ValueError("target bbox is clipped or outside the calibrated image")
    dw, dh = DEPTH_OUTPUT_SIZE
    decoder = zlib.decompressobj()
    raw = decoder.decompress(payload, dw * dh * 2 + 1)
    if not decoder.eof or decoder.unused_data or len(raw) != dw * dh * 2:
        raise ValueError("invalid Panther depth frame")
    depth = np.frombuffer(raw, dtype="<u2").reshape(dh, dw)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    u, v = min(dw - 1, int(cx * dw / width)), min(dh - 1, int(cy * dh / height))
    rx, ry = min(4, int((x2 - x1) * dw / width / 4)), min(4, int((y2 - y1) * dh / height / 4))
    if min(rx, ry) < 1:
        raise ValueError("target is too small for a depth sample")
    patch = depth[max(0, v - ry):v + ry + 1, max(0, u - rx):u + rx + 1]
    valid = patch[(patch >= 100) & (patch <= 3000)]
    if valid.size < 0.6 * patch.size or np.percentile(valid, 90) - np.percentile(valid, 10) > 45:
        raise ValueError("target center has missing or inconsistent depth")
    z = float(np.median(valid)) / 1000
    fx, fy, px, py = intrinsics
    point = (transform @ [(cx - px) * z / fx, (cy - py) * z / fy, z, 1.0])[:3]
    point[2] += surface_offset
    above = point + [0, 0, clearance]
    if np.any(np.minimum(point, above) < bounds[0]) or np.any(np.maximum(point, above) > bounds[1]):
        raise ValueError("pick or clearance point is outside the configured workspace")
    return point.tolist(), above.tolist()


class TabletopPick:
    def __init__(self, config, hub, motion, vop, *, notify=notify_async):
        self._config, self._hub, self._motion, self._vop = config, hub, motion, vop
        self._notify = notify
        self._lock = threading.Lock()
        self._started = False
        self._active = None
        self._last = None
        self._done = threading.Event()
        self._done.set()

    def get_tool(self):
        actions = ["start", "info", "preview", "pick", "stop"]
        return tool(CARD, "actuator",
                    "按现有 VOP 类别与可选颜色单次抓取桌面物体。preview 只计算抓取点；pick 抓起后保持夹爪闭合。",
                    {"type": "object", "properties": {
                        "action": {"type": "string", "enum": actions},
                        "object_name": {"type": "string", "maxLength": 128},
                        "color": {"type": "string", "maxLength": 64},
                        "confirm_motion": {"type": "boolean", "const": True}},
                     "required": ["action"],
                     "x-action-params": {
                         "preview": {"params": ["object_name", "color"]},
                         "pick": {"params": ["object_name", "color", "confirm_motion"]}},
                     "x-resource": ["base", "waist", "head", "arm_l", "arm_r", "hand_l", "hand_r"],
                     "x-completion": {"actions": ["pick"], "timeout": 120}},
                    topic_in=[{"format": "data/json", "desc": "同一 Panther camera_color 接入 VOP 后的 objects 流"}])

    def _topic(self):
        return self._hub.topics["camera_color"]["topic"] + "/objects"

    def _info(self):
        try:
            load_setup(self._config)
            setup = "configured"
        except ValueError as exc:
            setup = str(exc)
        return {"state": "ready" if self._started else "idle",
                "topic_in": [{**self.get_tool()["topic_in"][0], "topic": self._topic()}],
                "calibration": setup, "busy": self._active is not None, "last_result": self._last}

    def dispatch(self, action, args):
        try:
            with self._lock:
                if action == "info":
                    return self._info()
                if action == "stop":
                    return self._stop()
                if action == "start":
                    topics = args.get("input_topics") or [args.get("input_topic")]
                    if topics != [self._topic()]:
                        raise ValueError("connect only VOP objects produced from this Panther camera_color")
                    if self._active is not None:
                        raise RuntimeError("pick is still active")
                    self._vop.start(self._topic())
                    self._started = True
                    return self._info()
                if action not in ("preview", "pick"):
                    raise ValueError(f"unknown action {action!r}")
                if not self._started or self._active is not None:
                    raise RuntimeError("start tabletop_pick first and wait for any active pick to finish")
                if action == "pick":
                    require_confirmation(args)
                plan = self._preview(args)
                if action == "preview":
                    return {"state": "ready", **plan}
                cancel = threading.Event()
                self._motion.reserve_pick(cancel)
                action_id = f"panther_pick_{uuid4().hex[:8]}"
                self._active = cancel
                self._done.clear()
                try:
                    threading.Thread(target=self._run, args=(action_id, plan, cancel),
                                     name=CARD, daemon=True).start()
                except Exception:
                    self._active = None
                    self._done.set()
                    self._motion.release_pick(cancel)
                    raise
                return {"state": "picking", "action_id": action_id}
        except (ValueError, RuntimeError, TimeoutError, zlib.error) as exc:
            return {"state": "error", "message": str(exc)}

    def _preview(self, args):
        setup = load_setup(self._config)
        name, color = args.get("object_name"), args.get("color", "")
        if not isinstance(name, str) or not name.strip() or not isinstance(color, str):
            raise ValueError("object_name must match a VOP class; color is optional")
        name, color = name.strip(), color.strip()
        profiles = self._config.get("objects") or {}
        profile = profiles.get(name.casefold())
        if not isinstance(profile, dict):
            raise ValueError(f"no commissioned grasp profile for {name!r}")
        confidence = finite_number(self._config.get("min_confidence", 0.5), "min_confidence")
        offset = finite_number(profile.get("surface_to_tcp_z_m"), "surface_to_tcp_z_m")
        opened = finite_number(profile.get("gripper_open"), "gripper_open")
        closed = finite_number(profile.get("gripper_close"), "gripper_close")
        mode = profile.get("gripper_mode", "parallel")
        if not (0 < confidence <= 1 and abs(offset) <= 0.15 and 0 <= opened <= 1 and
                0 <= closed <= 1 and opened != closed and mode in GRIPPER_MODES):
            raise ValueError("invalid confidence, tool offset or gripper settings")
        self._guard(setup)
        data = self._vop.latest()
        if not data:
            raise ValueError("waiting for VOP results")
        obj = select_object(data, name, color, confidence)
        rgb, depth = self._hub.pick_frames(data.get("frame_sha256"), MAX_AGE_S, MAX_SKEW_S)
        if list(rgb["size"]) != setup[1] or rgb["frame"] != depth["frame"]:
            raise ValueError("RGB and registered depth must use the calibrated size and color optical frame")
        point, above = grasp_points(obj.get("bbox"), depth["data"], setup, offset)
        return {"object_name": name, "color": color, "bbox": obj["bbox"], "point_m": point, "above_m": above,
                "gripper_open": opened, "gripper_close": closed, "gripper_mode": mode,
                "observed_at": rgb["received"]}

    def _guard(self, setup):
        joints = self._hub.joint_positions(max_age_s=0.5) or {}
        lift = self._hub.lift_position()
        if (lift is None or abs(lift - setup[6]) > 0.005 or
                any(name not in joints or abs(joints[name] - value) > 0.015
                    for name, value in zip(HEAD_JOINTS, setup[5]))):
            raise RuntimeError("head or lift differs from the calibrated observation pose")
        if not self._hub.pick_base_still():
            raise RuntimeError("wait for fresh, stationary base odometry")

    def _run(self, action_id, plan, cancel):
        result = {"object_name": plan["object_name"], "point_m": plan["point_m"],
                  "grasp_checked": False, "stage": "plan_pregrasp"}
        status = "error"
        closed = False
        attempted_grip = False
        try:
            setup = load_setup(self._config)
            side = self._config["side"]

            def guard():
                if cancel.is_set():
                    raise RuntimeError("pick cancelled")
                self._guard(setup)

            def move(stage, point, before_send=None):
                result["stage"] = stage
                self._motion.pick_move(side, self._config, point, setup[4], guard, before_send)

            def check_target():
                with self._lock:
                    guard()
                    fresh = self._preview(plan)
                if np.linalg.norm(np.subtract(plan["point_m"], fresh["point_m"])) > 0.015:
                    raise RuntimeError("target moved since preview; request another pick")

            guard()
            if time.monotonic() - plan["observed_at"] > MAX_AGE_S:
                raise RuntimeError("observation expired before motion")
            self._motion.pick_gripper(side, plan["gripper_mode"], plan["gripper_open"], {"open"}, guard)
            move("pregrasp", plan["above_m"], check_target)
            move("descend", plan["point_m"])
            result["stage"] = "close"
            attempted_grip = True
            self._motion.pick_gripper(side, plan["gripper_mode"], plan["gripper_close"], {"closed", "partial"}, guard)
            closed = True
            move("lift", plan["above_m"])
            result["stage"] = "lifted"
            status = "completed"
        except Exception as exc:
            status = "cancelled" if cancel.is_set() else "error"
            result["message"] = str(exc)
        finally:
            result["may_be_holding"] = closed or attempted_grip
            self._motion.release_pick(cancel)
            with self._lock:
                self._active = None
                self._last = {"action_id": action_id, "status": status, **result}
                self._done.set()
            self._notify(action_id, status, result, CARD)

    def _stop(self):
        self._started = False
        if self._active is not None:
            self._active.set()
        self._vop.close()
        return {"state": "stopping" if self._active is not None else "idle"}

    def stop(self):
        with self._lock:
            self._stop()
        if not self._done.wait(24):
            print("[panther] tabletop_pick stop timed out during driver shutdown", flush=True)

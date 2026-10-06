#!/usr/bin/env python3
"""
Teleoperation of a 3-tube concentric tube robot (CTR) through an Octopus (Marlin) board.

Joint vector (always this order, SI units internally):
    q = [z_outer, z_middle, z_inner, th_outer, th_middle, th_inner]   (m, m, m, rad, rad, rad)

G-code axis mapping (edit GCODE_AXES if your wiring differs):
    X/Y/Z = translation of outer/middle/inner tube   (sent in mm)
    A/B/C = rotation    of outer/middle/inner tube   (sent in degrees)

Snap-through protection: SnapGuard limits the relative base rotation between tubes using a
reduced-order stability model (see its docstring). Calibrate it on the real robot.

Usage:
    python ctr-sim.py                    # TEST MODE (no hardware, prints G-code)
    python ctr-sim.py --live             # real hardware
    python ctr-sim.py --live --resume    # start from the pose saved at last clean exit
    python ctr-sim.py --live --port /dev/cu.usbmodem1101

Requires:  pip install numpy pygame pyserial     (NOT the package called "serial")
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pygame
import serial
import serial.tools.list_ports

from ctr_sim.tube import Tube
from ctr_sim.robot import ConcentricTubeRobot
from ctr_sim.material import Material
from ctr_sim.state import CTRState
from ctr_sim.control.resolved_rate import resolved_rate_step_v2
from ctr_sim.control.jacobian import numerical_position_jacobian_v2
from ctr_sim.mechanics.forward_v2 import solve_forward_kinematics_v2

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
GCODE_AXES = ("X", "Y", "Z", "A", "B", "C")

# Tube order everywhere: [outer, middle, inner]
# Single source of truth: builds the ctr_sim robot AND the snap-through guard.
# 'kappa' is passed to ctr_sim unchanged; the guard interprets it as curvature in 1/m.
TUBE_SPECS = [
    dict(name="outer_tube",  length=0.115, kappa=0.11051,  od=0.001534, id=0.0013208),
    dict(name="middle_tube", length=0.166, kappa=0.02177, od=0.00127, id=0.0010668),
    dict(name="inner_tube",  length=0.353, kappa=0.01964, od=0.00101092, id=0.0006604),
]
YOUNGS_MODULUS = 50e9        # Pa (Nitinol)
SHEAR_MODULUS = 20e9         # Pa
TUBE_LENGTHS = np.array([s["length"] for s in TUBE_SPECS])     # m

# --- Snap-through guard (see SnapGuard) ---
SNAP_SAFETY = 0.6                          # use at most this fraction of the predicted snap angle
SNAP_SOFT_ZONE = np.radians(30)            # start slowing down this far from the limit
FALLBACK_REL_LIMIT = np.radians(360)       # cap on relative base rotation when the model sees no bistability
INCLUDE_TRANSMISSION_TORSION = True        # also model twist of the straight tube length behind the base (real hardware;
                                           # ctr_sim itself assumes that part is torsionally rigid)
# Experimentally measured SAFE relative-rotation limits override the model, used as-is (rad):
#   MEASURED_REL_LIMIT = {(1, 2): np.radians(200)}      # pair = (outer-side tube, inner-side tube)
MEASURED_REL_LIMIT = {}
INITIAL_Q = np.array([0.006, 0.012, 0.017, 0.0, 0.0, 0.0])        # pose the tubes are physically in at start; order goes in outer tube, middle tube, inner tube. 1cm, 2cm, 3cm respectively

# Joint limits. Also clamp the translations to the real travel of your linear stages!
Q_MIN = np.concatenate([np.zeros(3), np.full(3, -2 * np.pi)])
Q_MAX = np.concatenate([TUBE_LENGTHS, np.full(3, 2 * np.pi)])

# Per-joint speed limits (m/s and rad/s). If any joint exceeds its limit, the whole
# joint step is scaled down so the tip still moves in the commanded direction.
MAX_JOINT_RATE = np.array([0.03, 0.03, 0.03, 1.0, 1.0, 1.0])

CONTROL_DT = 0.1             # s (10 Hz, same as the ctr_sim ROS node; a Jacobian refresh takes ~0.1 s)
JACOBIAN_REFRESH_TICKS = 3   # reuse the numerical Jacobian for this many Cartesian ticks (as in ctr_sim)
FEEDRATE = 3000              # mm/min for G1 (per-axis max feedrate/accel are set in firmware)
MAX_IN_FLIGHT = 3            # max G1 commands sent but not yet acknowledged ("ok") by Marlin

JOG_Z_SPEED = 0.01           # m/s   (joint-jog mode)
JOG_THETA_SPEED = 0.5        # rad/s (joint-jog mode)

DEADZONE = 0.15
RIGHT_STICK_Y_AXIS = 3       # gamepad dependent! Print joy.get_axis(i) to check yours.

STATE_FILE = "ctr_last_q.json"
CART, JOINT = "cartesian", "joint"


# ----------------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------------
class TeleopInput:
    """Keyboard + gamepad via pygame. The pygame window must have focus."""

    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        self.screen = pygame.display.set_mode((470, 280))
        pygame.display.set_caption("CTR Teleop")
        self.font = pygame.font.SysFont(None, 22)
        self.mode = CART
        self.selected = 0          # tube selected in joint-jog mode
        self.quit = False

        self.joy = None
        if pygame.joystick.get_count() > 0:
            self.joy = pygame.joystick.Joystick(0)
            print(f"Gamepad connected: {self.joy.get_name()}")
        else:
            print("No gamepad detected - keyboard only.")

    @staticmethod
    def _stick(v):
        """Deadzone with rescaling, so output ramps smoothly from 0 instead of jumping to 0.15."""
        if abs(v) < DEADZONE:
            return 0.0
        return float(np.sign(v) * (abs(v) - DEADZONE) / (1.0 - DEADZONE))

    def read(self):
        """Returns normalised [x, y, z] in [-1, 1]."""
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.quit = True
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    self.quit = True
                elif e.key == pygame.K_TAB:
                    self.mode = JOINT if self.mode == CART else CART
                elif e.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    self.selected = e.key - pygame.K_1

        k = pygame.key.get_pressed()
        x = float(k[pygame.K_d]) - float(k[pygame.K_a])
        y = float(k[pygame.K_UP]) - float(k[pygame.K_DOWN])
        z = float(k[pygame.K_w]) - float(k[pygame.K_s])

        if self.joy and self.mode == CART:
            x += self._stick(self.joy.get_axis(0))
            y -= self._stick(self.joy.get_axis(1))
            if self.joy.get_numaxes() > RIGHT_STICK_Y_AXIS:
                z -= self._stick(self.joy.get_axis(RIGHT_STICK_Y_AXIS))

        return np.clip([x, y, z], -1.0, 1.0)

    def draw(self, q, note="", snap_text="", snap_warn=False):
        self.screen.fill((25, 25, 30))
        if self.mode == CART:
            help_line = "CARTESIAN  W/S: Z   A/D: X   UP/DOWN: Y"
        else:
            help_line = f"JOINT JOG (tube {self.selected + 1})  W/S: translate   A/D: rotate"
        lines = ["[TAB] mode   [1/2/3] pick tube (1=outer)   [ESC] quit", help_line]
        for i, name in enumerate(("outer", "middle", "inner")):
            mark = ">" if (self.mode == JOINT and i == self.selected) else " "
            lines.append(f"{mark} {name:<6} z={q[i] * 1000:7.2f} mm   th={np.degrees(q[3 + i]):8.2f} deg")
        rows = [(t, (230, 230, 230)) for t in lines]
        if snap_text:
            rows.append((snap_text, (255, 120, 90) if snap_warn else (140, 220, 140)))
        if note:
            rows.append((note, (255, 200, 80)))
        for i, (text, color) in enumerate(rows):
            self.screen.blit(self.font.render(text, True, color), (12, 12 + 24 * i))
        pygame.display.flip()


# ----------------------------------------------------------------------------
# Hardware
# ----------------------------------------------------------------------------
class OctopusController:
    """
    Sends ABSOLUTE G1 moves to a Marlin board with 6 axes, with 'ok' flow control so the
    firmware queue never builds up (a deep queue = robot keeps moving after you let go).
    """

    def __init__(self, q0, port=None, baud=115200, live=False):
        self.live = live
        self.q = np.array(q0, dtype=float)    # last pose that was actually sent
        self.in_flight = 0
        self._rx = b""
        self.ser = None

        if not live:
            print("TEST MODE: commands are printed, nothing is sent.")
            return

        port = port or self._find_port()
        if port is None:
            raise ConnectionError("Could not find the Octopus board (try --port).")
        print(f"Connecting to Octopus on {port}")
        self.ser = serial.Serial(port, baud, timeout=0, write_timeout=1.0)

        time.sleep(2.5)                       # opening the port usually resets the board
        self.ser.reset_input_buffer()         # discard boot banner
        self._command_blocking("G90")                                   # absolute positioning
        self._command_blocking("G92 " + self._axis_words(self.q))       # firmware position := q0
        print("Firmware position synchronised to start pose.")

    @staticmethod
    def _find_port():
        for p in serial.tools.list_ports.comports():
            dev, desc = p.device.lower(), (p.description or "").upper()
            if "usbmodem" in dev or "ttyacm" in dev or "MARLIN" in desc:
                return p.device
        return None

    @staticmethod
    def _axis_words(q):
        vals = np.concatenate([q[:3] * 1000.0, np.degrees(q[3:])])      # m->mm, rad->deg
        return " ".join(f"{a}{v:.3f}" for a, v in zip(GCODE_AXES, vals))

    def _poll(self):
        """Non-blocking read; counts 'ok' acknowledgements, prints errors."""
        data = self.ser.read(self.ser.in_waiting)
        if not data:
            return
        self._rx += data
        while b"\n" in self._rx:
            line, self._rx = self._rx.split(b"\n", 1)
            text = line.decode(errors="replace").strip()
            if text.lower().startswith("ok"):
                self.in_flight = max(0, self.in_flight - 1)
            elif text.startswith(("Error", "!!")):
                print(f"[MARLIN] {text}")

    def _command_blocking(self, cmd, timeout=5.0):
        self.ser.write((cmd + "\n").encode())
        self.in_flight += 1
        deadline = time.monotonic() + timeout
        while self.in_flight > 0:
            self._poll()
            if time.monotonic() > deadline:
                raise TimeoutError(f"No 'ok' from firmware for {cmd!r}")
            time.sleep(0.005)

    def move_to(self, q_target):
        """Send an absolute pose. Returns False (and does nothing) if the firmware queue is busy."""
        cmd = f"G1 {self._axis_words(q_target)} F{FEEDRATE}"
        if self.live:
            self._poll()
            if self.in_flight >= MAX_IN_FLIGHT:
                return False
            self.ser.write((cmd + "\n").encode())
            self.in_flight += 1
        else:
            print(f"[VIRTUAL] {cmd}")
        self.q = np.array(q_target, dtype=float)
        return True

    def shutdown(self, emergency=False):
        if not self.live or self.ser is None:
            return
        try:
            if emergency:
                self.ser.write(b"M410\n")             # quick-stop: drop queued moves
            else:
                self._command_blocking("M400", timeout=10.0)   # let queued moves finish
        finally:
            self.ser.close()


# ----------------------------------------------------------------------------
# Robot model + kinematics wrapper
# ----------------------------------------------------------------------------
def setup_analytical_robot(q0):
    nitinol = Material(name="Nitinol", youngs_modulus=YOUNGS_MODULUS, shear_modulus=SHEAR_MODULUS)
    tubes = [Tube(name=s["name"], length=s["length"], precurvature=s["kappa"],
                  outer_diameter=s["od"], inner_diameter=s["id"], material=nitinol)
             for s in TUBE_SPECS]                                   # outer -> inner
    state = CTRState(insertions=q0[:3].tolist(), rotations=q0[3:].tolist())
    return ConcentricTubeRobot(tubes, state)


def sync_robot(robot, q):
    """Make the model match what was really commanded (also undoes any state the library may mutate)."""
    robot.state.insertions = q[:3].tolist()
    robot.state.rotations = q[3:].tolist()


class KinematicsCache:
    """
    Resolved-rate wrapper that mirrors ctr_sim's ROS node: warm-started torsion solves and Jacobian
    reuse. (A full numerical Jacobian costs ~0.1 s, so recomputing it every tick would stall the loop.)

    ctr_sim API used (verified against the repo):
        resolved_rate_step_v2(robot, dx, torsion_initial_guess=None, jacobian=None) -> dq = pinv(J) @ dx
        dx is a Cartesian DISPLACEMENT, dq a joint INCREMENT [d_beta (3), d_alpha (3)], gain 1.0.
    """

    def __init__(self):
        self.jacobian = None
        self.age = 0
        self.guess = None                     # base torsional strains theta_dot_i(0) of the last solve

    def invalidate(self):
        self.jacobian = None

    def _refresh(self, robot):
        last_exc = None
        for guess in (self.guess, None):      # warm start first, then a cold start
            try:
                backbone = solve_forward_kinematics_v2(robot, torsion_initial_guess=guess)
                self.guess = np.array(backbone.torsion_solution.base_theta_dot, dtype=float)
                self.jacobian = numerical_position_jacobian_v2(robot, torsion_initial_guess=self.guess)
                self.age = 0
                return
            except Exception as exc:          # ctr_sim's shooting solver is not guaranteed to converge
                last_exc = exc
        self.guess, self.jacobian = None, None
        raise RuntimeError(f"ctr_sim torsion solve did not converge at this pose ({last_exc})")

    def dq(self, robot, v, dt):
        """Joint increment (6,) for a desired tip velocity v (m/s) over one tick of length dt."""
        if self.jacobian is None or self.age >= JACOBIAN_REFRESH_TICKS:
            self._refresh(robot)
        dq = np.asarray(resolved_rate_step_v2(robot, v * dt, jacobian=self.jacobian), dtype=float).reshape(-1)
        self.age += 1
        if dq.shape != (6,) or not np.all(np.isfinite(dq)):
            raise ValueError(f"resolved_rate_step_v2 returned unusable result: {dq}")
        return dq


def limit_rates(dq, dt):
    scale = np.max(np.abs(dq) / dt / MAX_JOINT_RATE)
    return dq / scale if scale > 1.0 else dq


def jog_dq(u, tube, dt):
    dq = np.zeros(6)
    dq[tube] = u[2] * JOG_Z_SPEED * dt                 # W/S -> translate selected tube
    dq[3 + tube] = u[0] * JOG_THETA_SPEED * dt         # A/D -> rotate selected tube
    return dq


# ----------------------------------------------------------------------------
# Snap-through protection
# ----------------------------------------------------------------------------
# Tip relative angles psi used to trace the pair BVP (log-spaced near 0, where the solution is steep).
_PSI = np.concatenate([np.geomspace(1e-6, 0.05, 80), np.linspace(0.05, 2 * np.pi, 320)[1:]])


def _first_fold(alpha):
    """alpha(psi) = relative base angle needed to hold relative tip angle psi. The tip snaps where
    alpha first stops increasing (fold). Returns inf if the branch is monotonic (no snap)."""
    idx = np.nonzero(np.diff(alpha) < -1e-9)[0]
    return float(alpha[idx[0]]) if idx.size else np.inf


class SnapGuard:
    """
    Snap-through (elastic instability) guard for concentric precurved tubes.

    For an adjacent pair (a = outer side, b = inner side), the free-tip torsion problem over their
    overlap [0, Lc] reduces to a pendulum equation for the relative angle phi = theta_b - theta_a:
        phi'' = c * sin(phi),   c = kappa_a*kappa_b*(EI_a/GJ_b + EI_b/GJ_a),   phi'(Lc) = 0,
        phi(0) = relative angle at the base of the backbone.
    This is the same ODE ctr_sim integrates (torsion_v2.segment_torsion_ode). Shooting from the tip
    (phi = psi, phi' = 0) gives the base angle alpha(psi); the tip snaps at the first fold of alpha(psi).
    Straight tube length behind the base adds torsional compliance (ctr_sim ignores it):
        alpha_motor = phi(0) + |phi'(0)| * Teff,   Teff = (Ta/GJ_a + Tb/GJ_b) / (1/GJ_a + 1/GJ_b)
    The guard uses the earlier of the two folds (with / without transmission twist), re-evaluated every
    step because Lc and Teff depend on the insertions, and keeps |relative motor angle| below
    SNAP_SAFETY * fold. Non-adjacent pairs (outer/inner) get FALLBACK_REL_LIMIT.

    VALIDATION (against ctr_sim's solver, 2-tube robots, no transmission): predicted fold angles of
    326 / 229 deg matched the angles where its warm-started solve stops converging (326 / 230 deg), and
    a monotonic case (no fold) was correctly reported as stable.
    LIMITS: only pairwise interaction is modelled (a third precurved tube in the overlap is ignored, so
    the safety factor and the fallback cap carry that uncertainty); insertion convention is ctr_sim's
    (tube i occupies [beta_i - L_i, beta_i], base at s = 0). Calibrate with MEASURED_REL_LIMIT.
    """
    NAMES = ("outer", "middle", "inner")
    PAIRS = ((0, 1), (1, 2), (0, 2))

    def __init__(self, specs, youngs=YOUNGS_MODULUS, shear=SHEAR_MODULUS):
        od = np.array([s["od"] for s in specs])
        idm = np.array([s["id"] for s in specs])
        area_moment = np.pi / 64.0 * (od ** 4 - idm ** 4)
        self.EI = youngs * area_moment
        self.GJ = shear * 2.0 * area_moment                 # J = 2 I
        self.L = np.array([s["length"] for s in specs])
        self.kappa = np.array([s["kappa"] for s in specs])
        self._cache = {}

    def _folds(self, a, b, z):
        """(fold angle per ctr_sim model, fold angle incl. transmission twist) for pair (a, b), in rad."""
        za, zb = round(z[a] * 2000) / 2000, round(z[b] * 2000) / 2000       # 0.5 mm cache resolution
        key = (a, b, za, zb)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        Lc = min(za, zb)                                    # both tubes exposed here
        c = self.kappa[a] * self.kappa[b] * (self.EI[a] / self.GJ[b] + self.EI[b] / self.GJ[a])
        if Lc <= 1e-4 or c <= 0.0:
            result = (np.inf, np.inf)
        else:
            Ta, Tb = max(self.L[a] - za, 0.0), max(self.L[b] - zb, 0.0)
            Teff = (Ta / self.GJ[a] + Tb / self.GJ[b]) / (1.0 / self.GJ[a] + 1.0 / self.GJ[b])
            n = int(np.clip(10 * np.sqrt(c) * Lc, 100, 600))
            h = Lc / n
            y, v = _PSI.copy(), np.zeros_like(_PSI)         # integrate in tau = distance from the tip
            for _ in range(n):                              # RK4 for y'' = c sin(y)
                k1y, k1v = v, c * np.sin(y)
                k2y, k2v = v + 0.5 * h * k1v, c * np.sin(y + 0.5 * h * k1y)
                k3y, k3v = v + 0.5 * h * k2v, c * np.sin(y + 0.5 * h * k2y)
                k4y, k4v = v + h * k3v, c * np.sin(y + h * k3y)
                y = y + h / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y)
                v = v + h / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
            result = (_first_fold(y), _first_fold(y + v * Teff))
        if len(self._cache) > 5000:
            self._cache.clear()
        self._cache[key] = result
        return result

    def limit(self, pair, z):
        """Max allowed |theta_b - theta_a| (rad) for this pair at insertions z."""
        if pair in MEASURED_REL_LIMIT:
            return MEASURED_REL_LIMIT[pair]
        a, b = pair
        if b - a != 1:
            return FALLBACK_REL_LIMIT
        fold_sim, fold_hw = self._folds(a, b, z)
        fold = min(fold_sim, fold_hw) if INCLUDE_TRANSMISSION_TORSION else fold_sim
        return min(SNAP_SAFETY * fold, FALLBACK_REL_LIMIT)

    def margin(self, q):
        """Smallest remaining rotation margin (rad) over all pairs, and which pair it is. <0 = unsafe."""
        z, th = q[:3], q[3:]
        best, label = np.inf, ""
        for pair in self.PAIRS:
            a, b = pair
            m = self.limit(pair, z) - abs(th[b] - th[a])
            if m < best:
                best, label = m, f"{self.NAMES[a]}/{self.NAMES[b]}"
        return best, label

    def filter_step(self, q, q_target):
        """
        Returns (q_safe, state) with state in {"ok", "limited", "blocked"}.
        Slows down near the limit, scales the step back to the boundary if it would cross it, and
        blocks steps that make an already-violated pose worse. Steps that improve the margin
        (unwinding) are always allowed.
        """
        step = q_target - q
        m_now, _ = self.margin(q)

        if 0.0 <= m_now < SNAP_SOFT_ZONE and self.margin(q + step)[0] < m_now:
            step = step * max(0.15, m_now / SNAP_SOFT_ZONE)          # approaching the limit: slow down
            q_target = q + step

        m_new, _ = self.margin(q_target)
        if m_new >= 0.0 or m_new >= m_now:
            return q_target, "ok"
        if m_now < 0.0:
            return q.copy(), "blocked"                               # already outside and getting worse

        lo, hi = 0.0, 1.0                                            # bisect to the largest safe fraction
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            if self.margin(q + mid * step)[0] >= 0.0:
                lo = mid
            else:
                hi = mid
        return q + lo * step, "limited"

    def report(self, q):
        z = q[:3]
        fmt = lambda f: "none" if np.isinf(f) else f"{np.degrees(f):.0f} deg"
        print("Snap-through guard (pair BVP, same torsion ODE as ctr_sim):")
        for pair in self.PAIRS[:2]:
            a, b = pair
            fold_sim, fold_hw = self._folds(a, b, z)
            print(f"  {self.NAMES[a]}/{self.NAMES[b]}: snap at {fmt(fold_sim)} (ctr_sim model), "
                  f"{fmt(fold_hw)} (with transmission twist) -> limit +/-{np.degrees(self.limit(pair, z)):.0f} deg")
        m, label = self.margin(q)
        if m < 0:
            print(f"  WARNING: start pose already violates the guard ({label}). "
                  "Only moves that reduce the relative rotation will be accepted.")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="talk to real hardware (default: test mode)")
    ap.add_argument("--resume", action="store_true", help=f"start from pose saved in {STATE_FILE}")
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--max-speed", type=float, default=0.02, help="tip speed in m/s")
    return ap.parse_args()


def main():
    args = parse_args()

    q0 = INITIAL_Q.copy()
    if args.live:
        if args.resume and os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                q0 = np.array(json.load(f)["q"], dtype=float)
        print("\nStart pose that will be declared to the firmware (G92):")
        print(f"  insertions [mm]: {np.round(q0[:3] * 1000, 2)}   rotations [deg]: {np.round(np.degrees(q0[3:]), 2)}")
        if input("Are the tubes PHYSICALLY in this pose? Type 'yes' to continue: ").strip().lower() != "yes":
            print("Aborted.")
            return

    robot = setup_analytical_robot(q0)
    guard = SnapGuard(TUBE_SPECS)
    guard.report(q0)
    hw = OctopusController(q0, args.port, args.baud, live=args.live)
    teleop = TeleopInput()

    # Kinematics smoke test (catches wrong return shape / non-convergence before you drive anything)
    kin = KinematicsCache()
    try:
        kin.dq(robot, np.array([0.0, 0.0, 0.02]), CONTROL_DT)
    except Exception as exc:
        print(f"WARNING: Cartesian mode unavailable at the start pose ({exc}).\n"
              "         Joint-jog mode (TAB) still works.")
    sync_robot(robot, hw.q)

    print("System ready. Click the pygame window to focus keyboard input.")
    emergency = False
    note, last_warn = "", 0.0
    last = time.perf_counter()

    try:
        while not teleop.quit:
            tick = time.perf_counter()
            dt = min(tick - last, 2 * CONTROL_DT)
            last = tick

            u = teleop.read()
            dq = np.zeros(6)
            kin_failed = False
            try:
                if np.any(np.abs(u) > 1e-6):
                    if teleop.mode == CART:
                        dq = kin.dq(robot, u * args.max_speed, dt)
                    else:
                        dq = jog_dq(u, teleop.selected, dt)
                        kin.invalidate()                       # pose changes outside the Jacobian's assumptions
            except Exception as exc:                           # solver non-convergence, singularity, ...
                kin_failed = True
                if tick - last_warn > 1.0:
                    print(f"[KINEMATICS] {exc}")
                    last_warn = tick
                dq = np.zeros(6)

            note = "KINEMATICS FAILED - use joint mode (TAB)" if kin_failed else ""
            if np.any(dq != 0.0):
                dq = limit_rates(dq, dt)
                desired = hw.q + dq
                q_target = np.clip(desired, Q_MIN, Q_MAX)
                if np.any(np.abs(q_target - desired) > 1e-12):
                    note = "JOINT LIMIT reached"
                q_target, snap_state = guard.filter_step(hw.q, q_target)
                if snap_state == "blocked":
                    note = "SNAP GUARD: blocked - unwind rotations in joint mode (TAB)"
                elif snap_state == "limited":
                    note = "SNAP GUARD: motion limited"
                if np.any(np.abs(q_target - hw.q) > 1e-9):
                    if not hw.move_to(q_target):
                        note = "firmware busy - tick skipped"

            sync_robot(robot, hw.q)                            # model always follows the hardware pose
            snap_margin, snap_pair = guard.margin(hw.q)
            teleop.draw(hw.q, note,
                        f"snap margin {np.degrees(snap_margin):6.1f} deg  ({snap_pair})",
                        snap_warn=snap_margin < SNAP_SOFT_ZONE)

            spare = CONTROL_DT - (time.perf_counter() - tick)
            if spare > 0:
                time.sleep(spare)

    except KeyboardInterrupt:
        pass
    except Exception:
        emergency = True
        raise
    finally:
        hw.shutdown(emergency=emergency)
        if args.live:
            if emergency:
                if os.path.exists(STATE_FILE):
                    os.remove(STATE_FILE)
                print("Emergency stop: pose is no longer known - re-home the tubes before the next run.")
            else:
                with open(STATE_FILE, "w") as f:
                    json.dump({"q": hw.q.tolist()}, f)
                print(f"Final pose saved to {STATE_FILE} (use --resume next time).")
        pygame.quit()


if __name__ == "__main__":
    main()
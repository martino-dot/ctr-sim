import time
import sys
import numpy as np
import pygame
import serial
import serial.tools.list_ports

from ctr_sim.tube import Tube
from ctr_sim.robot import ConcentricTubeRobot
from ctr_sim.material import Material
from ctr_sim.state import CTRState
from ctr_sim.control.resolved_rate import resolved_rate_step_v2

class TeleopInput:
    """Reads live gamepad and keyboard inputs via pygame."""
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        
        # Pygame MUST have a display window in focus to capture keyboard events
        self.screen = pygame.display.set_mode((300, 150))
        pygame.display.set_caption("CTR Teleop Control")
        
        # Render some basic instructions to the window
        font = pygame.font.SysFont(None, 24)
        instructions = [
            "Keep this window in focus!",
            "W/S : Forward / Backward (Z)",
            "A/D : Left / Right (X)",
            "UP/DOWN : Up / Down (Y)",
            "ESC : Quit"
        ]
        for i, text in enumerate(instructions):
            img = font.render(text, True, (255, 255, 255))
            self.screen.blit(img, (20, 20 + (i * 25)))
        pygame.display.flip()
        
        if pygame.joystick.get_count() == 0:
            print("No gamepad detected! Keyboard control only.")
            self.joy = None
        else:
            self.joy = pygame.joystick.Joystick(0)
            self.joy.init()
            print(f"Gamepad connected: {self.joy.get_name()}")

    def get_velocity(self, max_speed=0.02):
        """Returns [dx, dy, dz] in m/s based on inputs."""
        # Process OS events so the window doesn't freeze
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
        
        dx, dy, dz = 0.0, 0.0, 0.0
        
        # 1. Keyboard Input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            pygame.quit()
            sys.exit(0)
            
        if keys[pygame.K_w]: dz += 1.0
        if keys[pygame.K_s]: dz -= 1.0
        if keys[pygame.K_a]: dx -= 1.0
        if keys[pygame.K_d]: dx += 1.0
        if keys[pygame.K_UP]: dy += 1.0
        if keys[pygame.K_DOWN]: dy -= 1.0
        
        # 2. Gamepad Input
        if self.joy:
            joy_dx = self.joy.get_axis(0)
            joy_dy = -self.joy.get_axis(1)
            joy_dz = -self.joy.get_axis(3)
            
            deadzone = 0.15
            if abs(joy_dx) > deadzone: dx += joy_dx
            if abs(joy_dy) > deadzone: dy += joy_dy
            if abs(joy_dz) > deadzone: dz += joy_dz
            
        # Clamp to [-1.0, 1.0] so pressing a key and the stick simultaneously doesn't double speed
        dx = np.clip(dx, -1.0, 1.0)
        dy = np.clip(dy, -1.0, 1.0)
        dz = np.clip(dz, -1.0, 1.0)
        
        return np.array([dx, dy, dz]) * max_speed

class OctopusController:
    def __init__(self, baud=115200, test_mode=False):
        self.test_mode = test_mode
        self.current_q = np.zeros(6, dtype=float)
        
        if self.test_mode:
            print("TEST MODE: Simulating commands.")
            return

        port = self._find_octopus_port()
        if port is None:
            raise ConnectionError("Could not find the Octopus board.")
            
        print(f"Connecting to Octopus on port: {port}")
        self.ser = serial.Serial(port, baud, timeout=0.1)

    def _find_octopus_port(self):
        ports = serial.tools.list_ports.comports()
        for p in ports:
            if "usbmodem" in p.device.lower() or "MARLIN" in p.description.upper():
                return p.device
        return None

    def send_velocity_command(self, q_dot, dt):
        self.current_q += (q_dot * dt)
        
        z_mm = self.current_q[0:3] * 1000.0
        theta_deg = np.degrees(self.current_q[3:6])
        
        cmd = f"G1 X{z_mm[0]:.3f} Y{z_mm[1]:.3f} Z{z_mm[2]:.3f} " \
              f"A{theta_deg[0]:.3f} B{theta_deg[1]:.3f} C{theta_deg[2]:.3f} F3000\n"
        
        if self.test_mode:
            if np.any(np.abs(q_dot) > 1e-5): 
                print(f"[VIRTUAL] -> {cmd.strip()}")
        else:
            self.ser.write(cmd.encode('utf-8'))
            
        return self.current_q.copy()

def setup_analytical_robot():
    nitinol = Material(name="Nitinol", youngs_modulus=50e9, shear_modulus=20e9) 
    
    tube1 = Tube(
        name="inner_tube",
        length=0.450, 
        precurvature=15.0, 
        outer_diameter=0.0010, 
        inner_diameter=0.0008,
        material=nitinol
    )
    
    tube2 = Tube(
        name="middle_tube",
        length=0.350, 
        precurvature=10.0, 
        outer_diameter=0.0014, 
        inner_diameter=0.0012,
        material=nitinol
    )
    
    tube3 = Tube(
        name="outer_tube",
        length=0.250, 
        precurvature=5.0, 
        outer_diameter=0.0018, 
        inner_diameter=0.0016,
        material=nitinol
    )
    
    initial_state = CTRState(
        insertions=[0.10, 0.20, 0.30], 
        rotations=[0.0, 0.0, 0.0]
    )
    
    return ConcentricTubeRobot([tube3, tube2, tube1], initial_state)

def main():
    analytical_robot = setup_analytical_robot()
    hardware = OctopusController(test_mode=True)
    input_handler = TeleopInput()
    
    hardware.current_q[0:3] = [0.10, 0.20, 0.30] 
    
    dt = 0.05
    print("System Ready. Click the Pygame window to focus keyboard inputs.")
    
    while True:
        start_time = time.time()
        
        v_desired = input_handler.get_velocity(max_speed=0.02)
        
        if np.any(np.abs(v_desired) > 1e-5):
            q_dot = resolved_rate_step_v2(robot=analytical_robot, dx=v_desired)
            hardware.send_velocity_command(q_dot, dt)
            
            analytical_robot.state.insertions = hardware.current_q[0:3].tolist()
            analytical_robot.state.rotations = hardware.current_q[3:6].tolist()
        
        elapsed = time.time() - start_time
        if elapsed < dt:
            time.sleep(dt - elapsed)

if __name__ == "__main__":
    main()

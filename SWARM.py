from controller import Supervisor
import math
import numpy as np

# --- INITIALIZATION ---
robot = Supervisor()
robot_node = robot.getSelf()
timestep = int(robot.getBasicTimeStep())

# Capture Initial Position at t=0
# This allows all robots to use the same controller regardless of where they start.
initial_pos = robot_node.getPosition()
init_x = initial_pos[0]
init_y = initial_pos[1]

# Capture Initial Orientation (Theta) at t=0
initial_rot = robot_node.getOrientation()
# Simple conversion from rotation matrix to 2D yaw
init_theta = math.atan2(initial_rot[0], initial_rot[1])

display = robot.getDevice("display")
lidar = robot.getDevice("laser")
lidar.enable(timestep)

imu = robot.getDevice('imu inertial_unit')
imu.enable(timestep)

lw = robot.getDevice('left wheel')
rw = robot.getDevice('right wheel')

lw.setPosition(float('inf'))
rw.setPosition(float('inf'))
lw.setVelocity(0)
rw.setVelocity(0)

lw_enc = robot.getDevice('left wheel sensor')
rw_enc = robot.getDevice('right wheel sensor')

lw_enc.enable(timestep)
rw_enc.enable(timestep)

# --- CONSTANTS & KALMAN MATRICES ---
WHEEL_RADIUS = 0.0975
WHEEL_BASE = 0.33
map_res = 0.02
map_size = 750
dt = timestep / 1000.0

# Start the State X at the captured initial position
X = np.array([[init_x], [init_y], [init_theta]]) 

P = np.eye(3) * 0.1 
Q = np.diag([0.01, 0.01, 0.005]) 
I = np.eye(3)

occupancy_grid = np.zeros((map_size, map_size), dtype=np.uint8)

prev_left_enc = 0.0
prev_right_enc = 0.0

is_escaping = False
escape_dir = 1.0

# ----------------- MAIN LOOP -----------------
while robot.step(timestep) != -1:
    
    # --- NOISE CONFIGURATION ---
    
    # slip_noise: Random wheel error.
    slip_noise = 0.02 
    
    # imu_noise: Electrical jitter in orientation.
    imu_noise = 0.02
    sensor_noise = 0.05
    
    # --- STEP 1: READ SENSORS ---
    range_image = lidar.getRangeImage()
    num_rays = len(range_image)
    
    c_lw = lw_enc.getValue()
    c_rq = rw_enc.getValue()
    
    if math.isnan(c_lw): 
        c_lw = 0.0
    if math.isnan(c_rq): 
        c_rq = 0.0
        
    measured_theta = imu.getRollPitchYaw()[2]
    
    # --- STEP 2: MOVEMENT LOGIC (RECOVERY-BASED) ---
    speed = 5.0
    front_slice = range_image[int(num_rays*0.35) : int(num_rays*0.65)]
    min_dist = min([d for d in front_slice if d > 0.05], default=5.0)
    
    if is_escaping:
        l_speed = speed * 0.6 * escape_dir
        r_speed = -speed * 0.6 * escape_dir
        if min_dist > 1.2: 
            is_escaping = False
    elif min_dist < 0.6:
        is_escaping = True
        l_dist = range_image[int(num_rays*0.1)]
        r_dist = range_image[int(num_rays*0.9)]
        if l_dist > r_dist:
            escape_dir = 1.0
        else:
            escape_dir = -1.0
        l_speed = speed * 0.6 * escape_dir
        r_speed = -speed * 0.6 * escape_dir
    else:
        l_speed = speed
        r_speed = speed
        
    lw.setVelocity(l_speed)
    rw.setVelocity(r_speed)

    # --- STEP 3: PREDICT (Odometry + Slip) ---
    slip_left = np.random.normal(1.0, slip_noise)
    slip_right = np.random.normal(1.0, slip_noise)
    
    d_left = (c_lw - prev_left_enc) * WHEEL_RADIUS * slip_left
    d_right = (c_rq - prev_right_enc) * WHEEL_RADIUS * slip_right
    
    d_center = (d_left + d_right) / 2.0
    d_theta = (d_right - d_left) / WHEEL_BASE
    
    v_angular = d_theta / dt

    X[0] += d_center * np.cos(X[2]) 
    X[1] += d_center * np.sin(X[2]) 
    X[2] += d_theta
    
    P = P + Q

    # --- STEP 4: SWARM UPDATE LOGIC ---
    
    # Condition: Robots see each other every 100 steps (Simulation Proxy)
    can_see_peer = (robot.getTime() % 4.0 < 0.1) # Syncs every 4 seconds
    
    # Current Ground Truth (What the peer robot would see)
    actual_pos = robot_node.getPosition() 
    robot_x_gt = actual_pos[0]
    robot_y_gt = actual_pos[1]
    
    if can_see_peer:
        # FULL UPDATE (X, Y, Theta)
        z_x = robot_x_gt + np.random.normal(0, sensor_noise) 
        z_y = robot_y_gt + np.random.normal(0, sensor_noise)
        z_theta = measured_theta + np.random.normal(0, imu_noise)
        
        Z = np.array([[z_x], [z_y], [z_theta]])
        H = np.eye(3) 
        R = np.diag([sensor_noise**2, sensor_noise**2, imu_noise**2])
        
    else:
        # PARTIAL UPDATE (Theta Only)
        z_theta = measured_theta + np.random.normal(0, imu_noise)
        
        Z = np.array([[z_theta]])
        H = np.array([[0, 0, 1]]) 
        R = np.array([[imu_noise**2]])

    # Kalman Correction Math
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    
    X = X + K @ (Z - (H @ X))
    P = (I - K @ H) @ P

    robot_x = X[0,0]
    robot_y = X[1,0]
    robot_theta = X[2,0]

    # --- STEP 5: MAPPING (STOP WHEN TURNING) ---
    is_turning = abs(v_angular) > 0.05 
    
    if not is_turning:
        fov = lidar.getFov()
        for i in range(0, num_rays, 2):
            dist = range_image[i]
            if 0.1 < dist < 5.0: 
                alpha = (fov / 2.0) - (i * fov / num_rays)
                obj_x = robot_x + dist * math.cos(robot_theta + alpha)
                obj_y = robot_y + dist * math.sin(robot_theta + alpha)
                
                # Global Arena is 15x15m
                idx_x = int((obj_x + 7.5) / map_res)
                idx_y = int((7.5 - obj_y) / map_res)
                
                if 0 <= idx_x < map_size and 0 <= idx_y < map_size:
                    occupancy_grid[idx_y, idx_x] = 255

    prev_left_enc = c_lw
    prev_right_enc = c_rq
    
    # --- STEP 6: DISPLAY ---
    map_display = np.dstack([occupancy_grid]*3 + [np.full_like(occupancy_grid, 255)]).astype(np.uint8)
    ir_map = display.imageNew(map_display.tobytes(), display.BGRA, map_size, map_size)
    if ir_map:
        display.imagePaste(ir_map, 0, 0, False)
        display.imageDelete(ir_map)
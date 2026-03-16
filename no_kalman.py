from controller import Supervisor
import math
import numpy as np

# --- Initialize ---
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
display = robot.getDevice("display")

lidar = robot.getDevice("laser")
lidar.enable(timestep)

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

# --- Constants ---
WHEEL_RADIUS = 0.0975
WHEEL_BASE = 0.33
map_res = 0.02
map_size = 750
dt = timestep / 1000.0

X = np.array([[0.0], [0.0], [0.0]]) 
occupancy_grid = np.zeros((map_size, map_size), dtype=np.uint8)

prev_left_enc = 0.0
prev_right_enc = 0.0

is_escaping = False
escape_dir = 1.0

# ----------------- MAIN LOOP (NO KALMAN) -----------------
while robot.step(timestep) != -1:
    
    # --- NOISE CONFIGURATION ---
    # slip_noise: Simulates the wheels losing grip on the floor.
    # - Increase (e.g., 0.05): The map will tilt and drift much faster.
    # - Decrease (e.g., 0.00): The map will be perfectly straight (unrealistic).
    slip_noise = 0.02
    
    # --- STEP 1: READ SENSORS ---
    range_image = lidar.getRangeImage()
    num_rays = len(range_image)
    c_lw = lw_enc.getValue()
    c_rq = rw_enc.getValue()
    
    if math.isnan(c_lw): 
        c_lw = 0.0
    if math.isnan(c_rq): 
        c_rq = 0.0
    
    # --- STEP 2: MOVEMENT LOGIC ---
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

    # --- STEP 3: ODOMETRY WITH SLIP ---
    slip_left = np.random.normal(1.0, slip_noise)
    slip_right = np.random.normal(1.0, slip_noise)
    
    d_left = (c_lw - prev_left_enc) * WHEEL_RADIUS * slip_left
    d_right = (c_rq - prev_right_enc) * WHEEL_RADIUS * slip_right
    
    d_center = (d_left + d_right) / 2.0
    d_theta = (d_right - d_left) / WHEEL_BASE
    
    v_angular = d_theta / dt

    # Update State (No Kalman correction)
    X[0] += d_center * np.cos(X[2]) 
    X[1] += d_center * np.sin(X[2]) 
    X[2] += d_theta

    robot_x = X[0,0]
    robot_y = X[1,0]
    robot_theta = X[2,0]

    # --- STEP 4: MAPPING (STOP WHEN TURNING) ---
    # We simply check if v_angular is very small. If it is high, we are turning.
    is_turning = abs(v_angular) > 0.05 
    
    if not is_turning:
        fov = lidar.getFov()
        for i in range(0, num_rays, 2):
            dist = range_image[i]
            if 0.1 < dist < 5.0: 
                alpha = (fov / 2.0) - (i * fov / num_rays)
                obj_x = robot_x + dist * math.cos(robot_theta + alpha)
                obj_y = robot_y + dist * math.sin(robot_theta + alpha)
                idx_x = int((obj_x + 7.5) / map_res)
                idx_y = int((7.5 - obj_y) / map_res)
                
                if 0 <= idx_x < map_size and 0 <= idx_y < map_size:
                    occupancy_grid[idx_y, idx_x] = 255

    prev_left_enc = c_lw
    prev_right_enc = c_rq
    
    # --- STEP 5: DISPLAY ---
    map_display = np.dstack([occupancy_grid]*3 + [np.full_like(occupancy_grid, 255)]).astype(np.uint8)
    ir_map = display.imageNew(map_display.tobytes(), display.BGRA, map_size, map_size)
    if ir_map:
        display.imagePaste(ir_map, 0, 0, False)
        display.imageDelete(ir_map)
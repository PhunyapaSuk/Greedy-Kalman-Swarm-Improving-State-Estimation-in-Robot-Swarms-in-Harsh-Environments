from controller import Supervisor
import math
import random
import numpy as np

# log_file = open("kalman.csv", "w")
# log_file.write("Time,Error\n") # Header for CSV

# --- Initialize ---
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

#nodess
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

# --- Constants & Kalman Matrices ---
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

# H Matrix for Result 2 (Observing ONLY Theta)
H = np.array([[0, 0, 1]]) 

occupancy_grid = np.zeros((map_size, map_size), dtype=np.uint8)

prev_left_enc = 0.0
prev_right_enc = 0.0

is_escaping = False
escape_dir = 1.0

# wandering logic constants
MAX_SPEED = 5.0
OBSTACLE_THRESHOLD = 0.8  # Distance to start avoiding
CLEARANCE_THRESHOLD = 1.5 # Distance to stop escaping
NOISE_FACTOR = 0.2        # How much "randomness" to add

#capture the map
saved = False

# ----------------- MAIN LOOP (KALMAN) -----------------
while robot.step(timestep) != -1:
    
    # --- NOISE CONFIGURATION ---
    # slip_noise: Wheel error. (Increase = faster map drifting).
    slip_noise = 0.02 
    
    # imu_noise: Sensor jitter. 
    imu_noise = 0.02
    sensor_noise = 0.02

    # Update R matrix dynamically using your imu_noise variable
    R = np.array([[imu_noise]])
    
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
    
    # --- STEP 2: MOVEMENT LOGIC ---
    left_zone = range_image[0 : int(num_rays*0.33)]
    mid_zone = range_image[int(num_rays*0.33) : int(num_rays*0.66)]
    right_zone = range_image[int(num_rays*0.66) : num_rays]
    
    # Get minimum distances for each zone
    min_left = min([d for d in left_zone if d > 0.05], default=5.0)
    min_mid = min([d for d in mid_zone if d > 0.05], default=5.0)
    min_right = min([d for d in right_zone if d > 0.05], default=5.0)
    
    if is_escaping:
        # Stay in escape mode until the front is very clear
        l_speed = MAX_SPEED * 0.5 * escape_dir
        r_speed = -MAX_SPEED * 0.5 * escape_dir
        
        # Only stop escaping when the front AND sides have breathing room
        if min_mid > CLEARANCE_THRESHOLD:
            is_escaping = False
    else:
        if min_mid < OBSTACLE_THRESHOLD:
            # 2. Obstacle detected! Determine direction based on side clearance
            is_escaping = True
            escape_dir = 1.0 if min_left > min_right else -1.0
            l_speed = 0
            r_speed = 0
        else:
            # 3. Smooth Wandering: Go forward but drift slightly away from closer side
            # This prevents the robot from getting perfectly parallel to walls
            steering_bias = (min_left - min_right) * 0.1 
            
            # Add a tiny bit of random "jitter" to break symmetry
            jitter = random.uniform(-NOISE_FACTOR, NOISE_FACTOR)
            
            l_speed = MAX_SPEED + steering_bias + jitter
            r_speed = MAX_SPEED - steering_bias - jitter
    
    # Final speed clamping
    l_speed = max(min(l_speed, MAX_SPEED), -MAX_SPEED)
    r_speed = max(min(r_speed, MAX_SPEED), -MAX_SPEED)
    
    lw.setVelocity(l_speed)
    rw.setVelocity(r_speed)

    # --- STEP 3: PREDICT (Odometry with Slip) ---
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

    # --- STEP 4: KALMAN MEASUREMENT & UPDATE ---
    # Apply your noise variables
    z_theta = measured_theta + np.random.normal(0, imu_noise)
    Z = np.array([[z_theta]])
   
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    
    X = X + K @ (Z - (H @ X))
    P = (I - K @ H) @ P

    robot_x = X[0,0]
    robot_y = X[1,0]
    robot_theta = X[2,0]

    # --- STEP 5: MAPPING (THRESHOLD-BASED) ---
    is_turning = abs(v_angular) > 0.05 
    
    # Logic: 
    # occupancy_grid tracks "hits". 
    # Starts at 0. Each hit adds 10.
    HIT_INC = 10       
    MAX_CONF = 100     
    THRESHOLD = 30     

    if not is_turning:
        fov = lidar.getFov()
        for i in range(0, num_rays, 2):
            dist = range_image[i]
            if 0.2 < dist < 3.5: 
                alpha = (fov / 2.0) - (i * fov / num_rays)
                obj_x = robot_x + dist * math.cos(robot_theta + alpha)
                obj_y = robot_y + dist * math.sin(robot_theta + alpha)
                
                # Center offset (adjust 7.5 based on your world size)
                idx_x = int((obj_x + 7.5) / map_res)
                idx_y = int((7.5 - obj_y) / map_res)
                
                if 0 <= idx_x < map_size and 0 <= idx_y < map_size:
                    # Increment confidence
                    current_val = int(occupancy_grid[idx_y, idx_x])
                    occupancy_grid[idx_y, idx_x] = min(current_val + HIT_INC, MAX_CONF)

    prev_left_enc = c_lw
    prev_right_enc = c_rq
    
    # --- STEP 6: DISPLAY ---
    # Logic: If confidence >= 30, color it WHITE (255). Otherwise, BLACK (0).
    display_pixels = np.where(occupancy_grid >= THRESHOLD, 255, 0).astype(np.uint8)
    
    # Stack into BGRA (Blue, Green, Red, Alpha)
    # This creates a grayscale image with a fully opaque alpha channel
    map_display = np.dstack([display_pixels, display_pixels, display_pixels, np.full_like(display_pixels, 255)])
    
    ir_map = display.imageNew(map_display.tobytes(), display.BGRA, map_size, map_size)
    if ir_map:
        display.imagePaste(ir_map, 0, 0, False)
        display.imageDelete(ir_map)
        
    if not saved and robot.getTime() >= 600:
        ref = display.imageCopy(0, 0, display.getWidth(), display.getHeight())
        display.imageSave(ref, "kalman_map.png")
        saved = True
        print("Image saved. Done.")
        
    # --- QUANTITATIVE DATA COLLECTION ---
    # current_time = robot.getTime()
    # if current_time > 600.0:
    #     print("600 seconds reached. Saving and Exiting...")
    #     log_file.close()
    #     # Optional: Save the final map automatically
    #     ref = display.imageCopy(0, 0, display.getWidth(), display.getHeight())
    #     display.imageSave(ref, "final_map_result.png")
    #     break 

    # # --- 3. LOGGING: Calculate and save error ---
    # actual_pos = robot_node.getPosition()
    # gt_x, gt_y = actual_pos[0], actual_pos[1]
    
    # # Euclidean Error between Kalman State X and Ground Truth
    # pos_error = math.sqrt((X[0,0] - gt_x)**2 + (X[1,0] - gt_y)**2)
    
    # # Write to file instead of printing
    # log_file.write(f"{current_time:.2f},{pos_error:.4f}\n")

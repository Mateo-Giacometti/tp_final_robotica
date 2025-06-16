#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import tf_transformations
import math
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

class Particle:
    def __init__(self, x, y, theta, weight, map_shape):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight
        self.log_odds_map = np.zeros(map_shape, dtype=np.float32)

    def pose(self):
        return np.array([self.x, self.y, self.theta])

class PythonSlamNode(Node):
    def __init__(self):
        super().__init__('python_slam_node')

        # Parameters
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        
        # TODO: define map resolution, width, height, and number of particles
        self.declare_parameter('map_resolution', 0.05)  # 5cm por celda
        self.declare_parameter('map_width_meters', 10.0)  # 10 metros de ancho
        self.declare_parameter('map_height_meters', 10.0)  # 10 metros de alto
        self.declare_parameter('num_particles', 50)  # Número de partículas

        self.resolution = self.get_parameter('map_resolution').get_parameter_value().double_value
        self.map_width_m = self.get_parameter('map_width_meters').get_parameter_value().double_value
        self.map_height_m = self.get_parameter('map_height_meters').get_parameter_value().double_value
        self.map_width_cells = int(self.map_width_m / self.resolution)
        self.map_height_cells = int(self.map_height_m / self.resolution)
        self.map_origin_x = -self.map_width_m / 2.0
        self.map_origin_y = -5.0

        # TODO: define the log-odds criteria for free and occupied cells
        self.declare_parameter('log_odds_occupied', 0.9)  # Incremento para celdas ocupadas
        self.declare_parameter('log_odds_free', -0.4)     # Decremento para celdas libres
        
        self.log_odds_occupied = self.get_parameter('log_odds_occupied').get_parameter_value().double_value
        self.log_odds_free = self.get_parameter('log_odds_free').get_parameter_value().double_value
        
        self.log_odds_max = 5.0
        self.log_odds_min = -5.0

        # Particle filter
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value
        self.particles = [Particle(0.0, 0.0, 0.0, 1.0/self.num_particles, (self.map_height_cells, self.map_width_cells)) for _ in range(self.num_particles)]
        self.last_odom = None
        
        # Variables para el estado actual
        self.current_map_pose = np.array([0.0, 0.0, 0.0])  # [x, y, theta] en el frame del mapa
        self.current_odom_pose = np.array([0.0, 0.0, 0.0])  # [x, y, theta] en el frame de odometría

        # ROS2 publishers/subscribers
        map_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.map_publisher = self.create_publisher(OccupancyGrid, '/map', map_qos_profile)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_subscriber = self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').get_parameter_value().string_value,
            self.odom_callback,
            10)
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            self.get_parameter('scan_topic').get_parameter_value().string_value,
            self.scan_callback,
            rclpy.qos.qos_profile_sensor_data)

        self.get_logger().info("Python SLAM node with particle filter initialized.")
        self.map_publish_timer = self.create_timer(1.0, self.publish_map)

    def odom_callback(self, msg: Odometry):
        # Store odometry for motion update
        self.last_odom = msg

    def scan_callback(self, msg: LaserScan):
        if self.last_odom is None:
            return

        # 1. Motion update (sample motion model)
        odom = self.last_odom
        # TODO: Retrieve odom_pose from odom message - remember that orientation is a quaternion
        odom_x = odom.pose.pose.position.x
        odom_y = odom.pose.pose.position.y
        odom_quat = odom.pose.pose.orientation
        _, _, odom_theta = tf_transformations.euler_from_quaternion([odom_quat.x, odom_quat.y, odom_quat.z, odom_quat.w])
        
        odom_pose = np.array([odom_x, odom_y, odom_theta])

        # TODO: Model the particles around the current pose
        for p in self.particles:
            # Add noise to simulate motion uncertainty
            # Ruido proporcional al movimiento
            motion_noise_x = np.random.normal(0, 0.05)  # 5cm de ruido en x
            motion_noise_y = np.random.normal(0, 0.05)  # 5cm de ruido en y
            motion_noise_theta = np.random.normal(0, 0.1)  # 0.1 rad de ruido angular
            
            p.x = odom_x + motion_noise_x
            p.y = odom_y + motion_noise_y
            p.theta = odom_theta + motion_noise_theta
            
            # Normalizar ángulo
            p.theta = self.normalize_angle(p.theta)

        # TODO: 2. Measurement update (weight particles)
        weights = []
        for p in self.particles:
            weight = self.compute_weight(p, msg) # Compute weights for each particle
            weights.append(weight)

        # Normalize weights
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(weights)) / len(weights)

        for i, p in enumerate(self.particles):
            p.weight = weights[i] # Resave weights

        # 3. Resample
        self.particles = self.resample_particles(self.particles)

        # TODO: 4. Use weighted mean of all particles for mapping and pose (update current_map_pose and current_odom_pose, for each particle)
        # Calcular la pose promedio ponderada
        total_weight = sum(p.weight for p in self.particles)
        if total_weight > 0:
            weighted_x = sum(p.x * p.weight for p in self.particles) / total_weight
            weighted_y = sum(p.y * p.weight for p in self.particles) / total_weight
            
            # Para ángulos, usar promedio circular
            sin_sum = sum(np.sin(p.theta) * p.weight for p in self.particles) / total_weight
            cos_sum = sum(np.cos(p.theta) * p.weight for p in self.particles) / total_weight
            weighted_theta = np.arctan2(sin_sum, cos_sum)
            
            self.current_map_pose = np.array([weighted_x, weighted_y, weighted_theta])
            self.current_odom_pose = odom_pose

        # 5. Mapping (update map with best particle's pose)
        for p in self.particles:
            self.update_map(p, msg)

        # 6. Broadcast map->odom transform
        self.broadcast_map_to_odom()

    def compute_weight(self, particle, scan_msg):
        # Simple likelihood: count how many endpoints match occupied cells
        score = 0.0
        robot_x, robot_y, robot_theta = particle.x, particle.y, particle.theta
        
        valid_measurements = 0
        
        for i, range_dist in enumerate(scan_msg.ranges):
            if range_dist < scan_msg.range_min or range_dist > scan_msg.range_max or math.isnan(range_dist):
                continue
                
            # TODO: Compute the map coordinates of the endpoint: transform the scan into the map frame
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            
            # Coordenadas del punto final del rayo en el marco del robot
            local_x = range_dist * np.cos(angle)
            local_y = range_dist * np.sin(angle)
            
            # Transformar al marco global usando la pose de la partícula
            global_x = robot_x + local_x * np.cos(robot_theta) - local_y * np.sin(robot_theta)
            global_y = robot_y + local_x * np.sin(robot_theta) + local_y * np.cos(robot_theta)
            
            # Convertir a coordenadas del mapa
            map_x = int((global_x - self.map_origin_x) / self.resolution)
            map_y = int((global_y - self.map_origin_y) / self.resolution)
            
            # Verificar que esté dentro del mapa
            if 0 <= map_x < self.map_width_cells and 0 <= map_y < self.map_height_cells:
                # TODO: Use particle.log_odds_map for scoring
                log_odds_value = particle.log_odds_map[map_y, map_x]
                
                # Si el punto debería estar ocupado y lo está en el mapa, incrementar score
                if range_dist < scan_msg.range_max:  # Hit
                    if log_odds_value > 0:  # Celda ocupada en el mapa
                        score += 1.0
                    else:
                        score -= 0.5  # Penalizar si no coincide
                
                valid_measurements += 1

        # Normalizar por el número de mediciones válidas
        if valid_measurements > 0:
            score = score / valid_measurements
        
        return max(score + 1e-6, 1e-6)  # Evitar pesos negativos o cero

    def resample_particles(self, particles):
        # TODO: Resample particles
        new_particles = []
        
        # Implementar resampling sistemático
        weights = [p.weight for p in particles]
        weights = np.array(weights)
        
        # Calcular índices para resampling
        indices = self.systematic_resample(weights)
        
        # Crear nuevas partículas basadas en los índices
        for idx in indices:
            old_particle = particles[idx]
            new_particle = Particle(
                old_particle.x, 
                old_particle.y, 
                old_particle.theta, 
                1.0 / len(particles),  # Peso uniforme después del resampling
                (self.map_height_cells, self.map_width_cells)
            )
            # Copiar el mapa
            new_particle.log_odds_map = old_particle.log_odds_map.copy()
            new_particles.append(new_particle)

        return new_particles
    
    def systematic_resample(self, weights):
        """Resampling sistemático para evitar degeneración de partículas"""
        N = len(weights)
        indices = []
        
        # Calcular suma acumulativa
        cumsum = np.cumsum(weights)
        
        # Generar puntos de muestreo sistemático
        u = np.random.uniform(0, 1/N)
        for i in range(N):
            u_i = u + i / N
            # Encontrar el índice correspondiente
            idx = np.searchsorted(cumsum, u_i)
            indices.append(min(idx, N-1))
        
        return indices

    def update_map(self, particle, scan_msg):
        robot_x, robot_y, robot_theta = particle.x, particle.y, particle.theta
        
        for i, range_dist in enumerate(scan_msg.ranges):
            is_hit = range_dist < scan_msg.range_max
            current_range = min(range_dist, scan_msg.range_max)
            if math.isnan(current_range) or current_range < scan_msg.range_min:
                continue
                
            # TODO: Update map: transform the scan into the map frame
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            
            # Coordenadas del punto final del rayo
            local_x = current_range * np.cos(angle)
            local_y = current_range * np.sin(angle)
            
            # Transformar al marco global
            global_x = robot_x + local_x * np.cos(robot_theta) - local_y * np.sin(robot_theta)
            global_y = robot_y + local_x * np.sin(robot_theta) + local_y * np.cos(robot_theta)
            
            # Coordenadas del robot en el mapa
            robot_map_x = int((robot_x - self.map_origin_x) / self.resolution)
            robot_map_y = int((robot_y - self.map_origin_y) / self.resolution)
            
            # Coordenadas del punto final en el mapa
            end_map_x = int((global_x - self.map_origin_x) / self.resolution)
            end_map_y = int((global_y - self.map_origin_y) / self.resolution)
            
            # Verificar límites del mapa
            if (0 <= robot_map_x < self.map_width_cells and 0 <= robot_map_y < self.map_height_cells and
                0 <= end_map_x < self.map_width_cells and 0 <= end_map_y < self.map_height_cells):
                
                # TODO: Use self.bresenham_line for free cells
                self.bresenham_line(particle, robot_map_x, robot_map_y, end_map_x, end_map_y)
                
                # TODO: Update particle.log_odds_map accordingly
                # Marcar el punto final como ocupado si es un hit válido
                if is_hit and range_dist < scan_msg.range_max:
                    particle.log_odds_map[end_map_y, end_map_x] += self.log_odds_occupied
                    particle.log_odds_map[end_map_y, end_map_x] = np.clip(
                        particle.log_odds_map[end_map_y, end_map_x], 
                        self.log_odds_min, 
                        self.log_odds_max
                    )

    def bresenham_line(self, particle, x0, y0, x1, y1):
        """Algoritmo de Bresenham para marcar celdas libres en la línea del rayo"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        path_len = 0
        max_path_len = dx + dy
        
        while not (x0 == x1 and y0 == y1) and path_len < max_path_len:
            if 0 <= x0 < self.map_width_cells and 0 <= y0 < self.map_height_cells:
                particle.log_odds_map[y0, x0] += self.log_odds_free
                particle.log_odds_map[y0, x0] = np.clip(particle.log_odds_map[y0, x0], self.log_odds_min, self.log_odds_max)
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
            path_len += 1

    def publish_map(self):
        # TODO: Fill in map_msg fields and publish one map
        map_msg = OccupancyGrid()
        
        # Header
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = self.get_parameter('map_frame').get_parameter_value().string_value
        
        # Información del mapa
        map_msg.info.resolution = self.resolution
        map_msg.info.width = self.map_width_cells
        map_msg.info.height = self.map_height_cells
        map_msg.info.origin.position.x = self.map_origin_x
        map_msg.info.origin.position.y = self.map_origin_y
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.w = 1.0
        
        # Crear el mapa combinado de todas las partículas (promedio ponderado)
        combined_map = np.zeros((self.map_height_cells, self.map_width_cells), dtype=np.float32)
        total_weight = sum(p.weight for p in self.particles)
        
        if total_weight > 0:
            for particle in self.particles:
                combined_map += particle.log_odds_map * particle.weight / total_weight
        
        # Convertir log-odds a probabilidades de ocupación (0-100, -1 para desconocido)
        map_data = []
        for y in range(self.map_height_cells):
            for x in range(self.map_width_cells):
                log_odds = combined_map[y, x]
                if abs(log_odds) < 0.1:  # Zona desconocida
                    occupancy = -1
                else:
                    # Convertir log-odds a probabilidad
                    prob = 1.0 / (1.0 + np.exp(-log_odds))
                    occupancy = int(prob * 100)
                    occupancy = max(0, min(100, occupancy))  # Clamp entre 0 y 100
                
                map_data.append(occupancy)
        
        map_msg.data = map_data

        self.map_publisher.publish(map_msg)
        self.get_logger().debug("Map published.")

    def broadcast_map_to_odom(self):
        # TODO: Broadcast map->odom transform
        t = TransformStamped()
        
        # Header
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.get_parameter('map_frame').get_parameter_value().string_value
        t.child_frame_id = self.get_parameter('odom_frame').get_parameter_value().string_value
        
        # Calcular la transformación map -> odom
        # Esta transformación corrige la deriva de la odometría
        map_to_odom_x = self.current_map_pose[0] - self.current_odom_pose[0]
        map_to_odom_y = self.current_map_pose[1] - self.current_odom_pose[1]
        map_to_odom_theta = self.angle_diff(self.current_map_pose[2], self.current_odom_pose[2])
        
        # Posición
        t.transform.translation.x = map_to_odom_x
        t.transform.translation.y = map_to_odom_y
        t.transform.translation.z = 0.0
        
        # Orientación (convertir de ángulo a cuaternión)
        quat = tf_transformations.quaternion_from_euler(0, 0, map_to_odom_theta)
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        
        self.tf_broadcaster.sendTransform(t)

    @staticmethod
    def angle_diff(a, b):
        """Calcular la diferencia entre dos ángulos manteniendo el resultado en [-π, π]"""
        d = a - b
        while d > np.pi:
            d -= 2 * np.pi
        while d < -np.pi:
            d += 2 * np.pi
        return d
    
    @staticmethod
    def normalize_angle(angle):
        """Normalizar un ángulo al rango [-π, π]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    node = PythonSlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
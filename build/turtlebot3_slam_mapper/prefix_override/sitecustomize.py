import sys
if sys.prefix == '/home/mateo/miniconda3/envs/rosenv':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/mateo/Desktop/tp_final_robotica/install/turtlebot3_slam_mapper'

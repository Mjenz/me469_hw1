import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

class PotentialFields():
    """
    Potentaial fields for navigating area with obstacles to goal position.
    """
    def __init__(self,goal):
        """
        Initialize class instantiation of potential fields

        Create function in 3D space based on obstacle and goal locations.

        Loads in obstacle locations from ds1.

        Uses gaussian distributions centered at points to create symbolic function.

        ARGUMENTS:
        goal (array): coordinates of goal position

        """
        # initialize velocities and configuration
        self.v_vel = 0.0
        self.omega_vel = 0.0
        self.x = 0.0
        self.y = 0.0
        self.theta = -np.pi/2
        self.dt = 0.1

        # initiailze maximum accelerations and velocities
        self.v_dot_max = 0.288
        self.omega_dot_max = 5.579
        self.v_vel_max = 10.0
        self.v_vel_min = 0.0
        self.omega_vel_max = 1.0 
        self.omega_vel_min = -self.omega_vel_max 

        # create sympy symbols
        self.x_sym = sp.Symbol('x')
        self.y_sym = sp.Symbol('y')

        # intialize potential function
        U = 0.0

        # define obstacle amplitude and st_dev for obstacles
        a = -1000.0
        sigma = 0.5

        # load obstacle location
        obstacles = np.loadtxt('ds1_Landmark_Groundtruth.dat')

        # create potential function
        for obs in obstacles:
            # unpack x and y
            x_obs = obs[1]
            y_obs = obs[2]
            
            U += (a * sp.exp(-((self.x_sym-x_obs)**2 + (self.y_sym-y_obs)**2)/(2 * sigma**2)))
        
        # intialize amplitude and st_dev for goal
        a1 = 10000.0
        sigma1 = 10.0
        a2 = 100.0
        sigma2 = 1
        # add goal to potential function
        U += (a1 * sp.exp(-((self.x_sym-goal[0])**2 + (self.y_sym-goal[1])**2)/(2 * sigma1**2)))
        U += (a2 * sp.exp(-((self.x_sym-goal[0])**2 + (self.y_sym-goal[1])**2)/(2 * sigma2**2)))

        # take partial derivatives of function
        d_U_x = sp.diff(U,self.x_sym)
        d_U_y = sp.diff(U,self.y_sym)

        # convert all three symbolic functions to numerical functions
        self.U_lam = sp.lambdify([self.x_sym,self.y_sym],U)
        self.d_U_x_lam = sp.lambdify([self.x_sym,self.y_sym],d_U_x)
        self.d_U_y_lam = sp.lambdify([self.x_sym,self.y_sym],d_U_y)
        
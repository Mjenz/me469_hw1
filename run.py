import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from enum import auto, Enum



plt.style.use('config.mplstyle')
class State(Enum):
    LEFT = auto(),
    RIGHT = auto(),
    FORWARD = auto(),
    STOPPED = auto()

class Grid():
    """
        Class for representing occupation cell space for A star algorithm
    """

    def __init__(self,x_range,y_range,scale,pt=False):
        """
        Initialize a Grid class object.

        Represents a grid with unoccupied and occupied cells.

        PARAMETERS:
        x_range (list): grid x_min and x_max
        y_range (list): grid y_min and y_max
        scale (float): length of cell side in meters
        """
        # save arguments
        self.x_min = x_range[0]
        self.x_max = x_range[1]
        self.y_min = y_range[0]
        self.y_max = y_range[1]
        self.scale = scale
        self.prt = pt
        
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
        
        # intialize the grid of zeros
        self.grid = np.zeros([(x_range[1]-x_range[0]) * int(1/self.scale),((y_range[1]-y_range[0]) * int(1/self.scale))])

        # initialize labels for each axis. columns correspond to y and rows to x for ease of use purposes
        self.col_label = [y_range[0] + ii * self.scale for ii in range(np.shape(self.grid)[1]+1)]
        self.row_label = [x_range[0] + jj * self.scale for jj in range(np.shape(self.grid)[0]+1)]

        # load obstacle location
        obstacles = np.loadtxt('ds1_Landmark_Groundtruth.dat')

        # initialize array of occupied indexes
        self.occupied_cells = []

        # loop through list of obstacle coordinates
        for obs in obstacles:
            # unpack x and y
            x = obs[1]
            y = obs[2]
            # loop through all x values in grid
            for ii in range(len(self.row_label)-1):
                # if the x location of the occupied cell is between the bounds
                if x >= self.row_label[ii] and x < self.row_label[ii+1]:
                    # loop through all y values in grid
                    for jj in range(len(self.col_label)-1):
                        # if the y location of the occupied cell is between the bounds
                        if y >= self.col_label[jj] and y < self.col_label[jj+1]:
                            # save the cell indicies as an occupied cell
                            self.occupied_cells.append([ii,jj]) # rows, cols (y, x)

        # expand object markers if using smaller cell size
        if self.scale == 0.1:
            # save list of initially occupied cells (non-expanded)
            initial_occupied = self.occupied_cells.copy()
            # loop through all known obstacles
            for cell in initial_occupied:
                # unpack x and y indexes
                ii  = cell[0]
                jj  = cell[1]
                # loop through x values in cells +-3 away from center cell
                for i in range(ii-3,ii+3):
                    # loop through y values in cells +-3 away from center cell
                    for j in range(jj-3,jj+3):
                        # ignore the center cell (it is already marked as occupied)
                        if ii == i and jj == j:
                            pass
                        else:
                            # save the neighbor cells as occupied
                            self.occupied_cells.append([i,j]) # rows, cols (y, x)
        
        # loop through indexes of known occupied cells
        for cell in self.occupied_cells:
            # mark as occupied on the grid
            self.grid[cell[0],cell[1]] = 1   

        # if printing is enabled, print
        if self.prt:
            print(f'grid {np.shape(self.grid)}:\n{self.grid}\n')
            print(f'row_label {np.shape(self.row_label)}:\n{self.row_label}\n')
            print(f'col_label {np.shape(self.col_label)}:\n{self.col_label}\n')
            print(f'occupied_cells {np.shape(self.occupied_cells)}:\n{self.occupied_cells}\n')
            print(f'grid {np.shape(self.grid)}:\n{self.grid}\n')

    def crdnt_to_idx(self,coordinate):
        """
        Converts point coordinates to grid cell indexes
        [x,y] --> [ii,jj]

        PARAMETERS:
        coordinates (list): the x and y coordinate pair corresponding to the grid cell location

        RETURNS:
        index (list): the corresponding row column pair

        """
        for ii in range(len(self.row_label)-1):
            if coordinate[0] >= self.row_label[ii] and coordinate[0] < self.row_label[ii+1]:
                for jj in range(len(self.col_label)-1):
                    if coordinate[1] >= self.col_label[jj] and coordinate[1] < self.col_label[jj+1]:
                        return [ii,jj]

    def idx_to_crdnt(self,index):
        """
        Converts grid cell indexes to point coordinates (estimated)
        [ii,jj] --> [x,y] 

        PARAMETERS:
        index(list): the row column pair to be converted

        RETURNS:
        coordinates (list): the x and y coordinate pair corresponding to the grid cell location
        """

        return [index[0] * self.scale + self.x_min, index[1] * self.scale + self.y_min]
    
    def get_neighbors(self,index):
        """
        Converts grid cell indexes to point coordinates (estimated)
        [ii,jj] --> [x,y] 
        """
        n = []

        x_inc = 0 if (index[0] + 1) >=  np.shape(self.grid)[0] else 1
        x_dec = 0 if (index[0] - 1) <   0 else 1
        y_inc = 0 if (index[1] + 1) >=  np.shape(self.grid)[1] else 1
        y_dec = 0 if (index[1] - 1) <   0 else 1

        if x_inc:
            n.append([index[0]+1,index[1]])
        if x_dec:
            n.append([index[0]-1,index[1]])
        if y_inc:
            n.append([index[0],index[1]+1])
        if y_dec:
            n.append([index[0],index[1]-1])
        if x_inc and y_inc:
            n.append([index[0]+1,index[1]+1])
        if x_dec and y_dec:
            n.append([index[0]-1,index[1]-1])
        if x_inc and y_dec:
            n.append([index[0]+1,index[1]-1])
        if x_dec and y_inc:
            n.append([index[0]-1,index[1]+1])
            
        return n

    def codeA_part2_Offline(self,start,goal):
        """
        Offline A* algorithm implementation

        Produces a path from start to goal

        PARAMETERS:
        start (list): x,y coordinates of start position
        goal (list): x,y coordinates of goal position

        OUTPUTS:
        self.path (list): list of cells along path from start to finish including start and goal cells
        """
        # save start and end position, convert to row col indexes
        self.start = start
        self.goal = goal
        self.start_index = self.crdnt_to_idx(start)
        self.goal_index = self.crdnt_to_idx(goal)
        goal_crdnt_center = np.array(self.idx_to_crdnt(self.goal_index)) + self.scale/2

        # if printing is enabled, print
        if self.prt:
            print(f'start {np.shape(start)}:\n{start}\n')
            print(f'goal {np.shape(goal)}:\n{goal}\n')
            print(f'start_index {np.shape(self.start_index)}:\n{self.start_index}\n')
            print(f'goal_index {np.shape(self.goal_index)}:\n{self.goal_index}\n')

        # instantiate open set with start row col index as part of open set
        open_set = [self.start_index]
        open_set_id = [0]
        # initialize total cost array and list for nodes in path
        f_list = []
        node_list = []
        node_id = []
        node_parent_id = []
        num = 0
        # loop
        while(1):
            # loop through open set
            for cell in open_set:
                # calculate heuristic (distance function in grid plane)
                crdnt = self.idx_to_crdnt(cell)
                crdnt = np.array(crdnt) + self.scale/2
                # h = np.sqrt(np.square(cell[0]-self.goal_index[0])+np.square(cell[1]-self.goal_index[1])) # this one works better
                h = np.sqrt(np.square(crdnt[0]-goal_crdnt_center[0])+np.square(crdnt[1]-goal_crdnt_center[1])) # this one works better                # true cost, provided by assignment
                if cell in self.occupied_cells:
                    g = 1000  # if it is an occupied cell
                else:
                    g = 1

                # calculate total cost and save
                f_list.append(g + h)
            
            # find lowest cost point
            min_idx = np.argmin(f_list)

            # clear f_list
            f_list = []

            # check if goal is reached
            if open_set[min_idx][0] == self.goal_index[0] and open_set[min_idx][1] == self.goal_index[1]:
                node_list.append(open_set.pop(min_idx))
                node_parent_id.append(open_set_id.pop(min_idx))
                node_id.append(num)
                break

            if len(open_set_id) != len(open_set) or len(node_id) != len(node_list) or len(node_parent_id) != len(node_id):
                raise
        
            # get neighbors of lowest cost cell
            n = self.get_neighbors(open_set[min_idx])

            # remove lowest cost cell from open set
            node_list.append(open_set.pop(min_idx))
            node_parent_id.append(open_set_id.pop(min_idx))
            node_id.append(num)

            # add neighbors to open set, if they are not in already
            for p in n:
                # check and make sure it is not already in the open set or that it is in the path
                if p not in open_set and p not in node_list:
                    # add to open set
                    open_set.append(p)
                    open_set_id.append(num)

            # num increase
            num += 1
            
            # visualize at every step if you would like (optional)
            # self.visualize_path()

        # save path to object
        self.plan = [node_list[-1]]
        next_up_id = node_parent_id[-1]

        # find the best path
        while(1):
            up_id = node_id.index(next_up_id)
            self.plan.append(node_list[up_id]) # add good path to plan list
            if up_id == 0:
                break
            next_up_id = node_parent_id[up_id]

    def codeA_part4_Online(self,start,goal):
        """
        Online A* algorithm implementation

        Produces a path from start to goal

        PARAMETERS:
        start (list): x,y coordinates of start position
        goal (list): x,y coordinates of goal position

        OUTPUTS:
        self.path (list): list of cells along path from start to finish including start and goal cells
        """
        # save start and end position, convert to row col indexes
        self.start = start
        self.goal = goal
        self.start_index = self.crdnt_to_idx(start)
        self.goal_index = self.crdnt_to_idx(goal)
        goal_crdnt_center = np.array(self.idx_to_crdnt(self.goal_index)) + self.scale/2

        # if printing is enabled, print
        if self.prt:
            print(f'start {np.shape(start)}:\n{start}\n')
            print(f'goal {np.shape(goal)}:\n{goal}\n')
            print(f'start_index {np.shape(self.start_index)}:\n{self.start_index}\n')
            print(f'goal_index {np.shape(self.goal_index)}:\n{self.goal_index}\n')

        # instantiate open set with start row col index as part of open set
        open_set = [self.start_index]

        # initialize total cost array and list for nodes in path
        f_list = []
        path_list = []

        # loop
        while(1):
            # loop through open set
            for cell in open_set:
                # calculate heuristic (distance function in grid plane)
                crdnt = self.idx_to_crdnt(cell)
                crdnt = np.array(crdnt) + self.scale/2
                # h = np.sqrt(np.square(cell[0]-self.goal_index[0])+np.square(cell[1]-self.goal_index[1])) # this one works better
                h = np.sqrt(np.square(crdnt[0]-goal_crdnt_center[0])+np.square(crdnt[1]-goal_crdnt_center[1])) # this one works better                # true cost, provided by assignment
                if cell in self.occupied_cells:
                    g = 1000  # if it is an occupied cell
                else:
                    g = 1

                # calculate total cost and save
                f_list.append(g + h)
            
            # find lowest cost point
            min_idx = np.argmin(f_list)

            # clear f_list
            f_list = []

            # check if goal is reached
            if open_set[min_idx][0] == self.goal_index[0] and open_set[min_idx][1] == self.goal_index[1]:
                # if np.isclose(open_set[min_idx][0],self.goal_index[0]) and np.isclose(open_set[min_idx][1] , self.goal_index[1]):
                path_list.append(open_set[min_idx]) # add goal cell to path list
                break

            # get neighbors of lowest cost cell
            n = self.get_neighbors(open_set[min_idx])

            # remove lowest cost cell from open set
            path_list.append(open_set.pop(min_idx))
            self.plan = path_list

            # add neighbors to open set, if they are not in already
            for p in n:
                # check and make sure it is not already in the open set or that it is in the path
                if p not in open_set and p not in path_list:
                    # add to open set
                    open_set.append(p)

            # visualize at every step if you would like (optional)
            # self.visualize_path()

        # save path to object
        self.plan = path_list

    def visualize_grid(self,name):
        """
        Visualize grid with obstacles
        """
        lin_x = np.linspace(self.x_min,self.x_max,10)
        lin_y = np.linspace(self.y_min,self.y_max,10)

        fig, ax = plt.subplots()
        for r in self.row_label:
            x = r * np.ones([10])
            ax.plot(x,lin_y,'-k',linewidth=0.5)

        for c in self.col_label:
            y = c * np.ones([10])
            ax.plot(lin_x,y,'-k',linewidth=0.5)

        for cell in self.occupied_cells:
            ################### https://www.geeksforgeeks.org/python/how-to-draw-shapes-in-matplotlib-with-python/ ############
            rect = Rectangle([self.row_label[cell[0]],self.col_label[cell[1]]],self.scale,self.scale,edgecolor='black',facecolor='black')
            ax.add_patch(rect)

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(name)
        fig.set_size_inches(5,7)
        plt.show()    

    def visualize_path(self,name):
        """
        Visualize grid with obstacles, path, and start and finish cells marked
        """
        # create points for plotting edges
        lin_x = np.linspace(self.x_min,self.x_max,10)
        lin_y = np.linspace(self.y_min,self.y_max,10)

        # create figure
        fig, ax = plt.subplots(figsize=[7,12])
        
        # draw lines
        for r in self.row_label:
            x = r * np.ones([10])
            ax.plot(x,lin_y,'-k',linewidth=0.5)

        for c in self.col_label:
            y = c * np.ones([10])
            ax.plot(lin_x,y,'-k',linewidth=0.5)

        # draw occupied cells
        for cell in self.occupied_cells:
            ################### https://www.geeksforgeeks.org/python/how-to-draw-shapes-in-matplotlib-with-python/ ############
            rect = Rectangle([self.row_label[cell[0]],self.col_label[cell[1]]],self.scale,self.scale,edgecolor='black',facecolor='black')
            ax.add_patch(rect)
        # again once for the legend label
        rect = Rectangle([self.row_label[cell[0]],self.col_label[cell[1]]],self.scale,self.scale,edgecolor='black',facecolor='black',label='obstacle')
        ax.add_patch(rect)

        # draw path cells
        for cell in self.plan:
            rect = Rectangle([self.row_label[cell[0]],self.col_label[cell[1]]],self.scale,self.scale,edgecolor='red',facecolor='red')
            ax.add_patch(rect)
        # again once for the legend label
        rect = Rectangle([self.row_label[cell[0]],self.col_label[cell[1]]],self.scale,self.scale,edgecolor='red',facecolor='red',label='path')
        ax.add_patch(rect)

        # draw goal cell 
        rect = Rectangle(self.idx_to_crdnt(self.goal_index),self.scale,self.scale,edgecolor='yellow',facecolor='yellow',label='goal')
        ax.add_patch(rect)

        # draw start cell
        rect = Rectangle(self.idx_to_crdnt(self.start_index),self.scale,self.scale,edgecolor='green',facecolor='green',label='start')
        ax.add_patch(rect)

        if self.scale == 0.1:
            print("hihi")
            # fit graph to area of path
            corr = np.array([self.idx_to_crdnt(cell) for cell in self.plan])
            print(corr)
            xlim = np.array([min(corr[:,0])-self.scale*2,max(corr[:,0])+self.scale*2])
            ylim = np.array([min(corr[:,1])-self.scale*2,max(corr[:,1])+self.scale*2])
            print(xlim,ylim)
            w = (xlim[1]-xlim[0])
            l = (ylim[1]-ylim[0])
            ratio = w/l
            desired_ratio = 5/7
            if np.isclose(ratio,desired_ratio,atol=0.25):
                pass
            elif ratio > desired_ratio:
                new_length = ratio/(desired_ratio) * l
                dif = new_length - l
                ylim[0] += dif
                ylim[1] -= dif
                ylim = [min(ylim),max(ylim)]
            elif ratio < desired_ratio:
                new_width = ratio/(desired_ratio) * w
                dif = new_width - w
                xlim[0] += dif
                xlim[1] -= dif

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(name)
        ax.legend()
        fig.set_size_inches(5,7)

        # plt.savefig(name, dpi=1000)
        plt.show()

        ########### CODE B ###################
    def find_if_cardinal_neighbors_occupied(self,index):
        """
        Converts grid cell indexes to point coordinates (estimated)
        [ii,jj] --> [x,y] 
        """
        n = []
        occupied = []
        # this does not work for all sclaes!!!! use grid size?
        x_inc = 0 if (index[0] + 1) >=  np.shape(self.grid)[0] else 1
        x_dec = 0 if (index[0] - 1) <   0 else 1
        y_inc = 0 if (index[1] + 1) >=  np.shape(self.grid)[1] else 1
        y_dec = 0 if (index[1] - 1) <   0 else 1

        if x_inc:
            n.append([index[0]+1,index[1]])
        if y_inc:
            n.append([index[0],index[1]+1])
        if x_dec:
            n.append([index[0]-1,index[1]])
        if y_dec:
            n.append([index[0],index[1]-1])

        for cell in n:
            if cell in self.occupied_cells:
                occupied.append(1)
            else:
                occupied.append(0)

        return n, occupied

    def codeB_prt8_controller(self, goal, position, velocity):
        # unpack parameters
        goal_x = goal[0]
        goal_y = goal[1]
        actual_x = position[0]
        actual_y = position[1]
        actual_theta = position[2]
        v_vel = velocity[0]
        omega_vel = velocity[1]

        distance_error = np.sqrt((goal_x - actual_x)**2 + (goal_y - actual_y)**2)
        heading_error = np.atan2((goal_y - actual_y),(goal_x - actual_x)) - actual_theta

        if abs(heading_error) > 3.14:
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        # controller
        if abs(heading_error) < 0.1:
            ideal_v_vel = State.FORWARD
        else:
            ideal_v_vel = State.STOPPED

        if heading_error > 0.0 and abs(heading_error) > 0.1:
            ideal_omega_vel = State.LEFT
        else:
            ideal_omega_vel = State.RIGHT
        
        # compute command velocities
        if ideal_v_vel == State.FORWARD:
            v_vel += self.v_dot_max * self.dt
        elif ideal_v_vel == State.STOPPED:
            v_vel += -self.v_dot_max * self.dt
        
        if ideal_omega_vel == State.LEFT:
            omega_vel += self.omega_dot_max * self.dt
        elif ideal_omega_vel == State.RIGHT:
            omega_vel += -self.omega_dot_max * self.dt

        # clamping
        if v_vel > self.v_vel_max:
            v_vel = self.v_vel_max
        elif v_vel < self.v_vel_min:
            v_vel = self.v_vel_min

        if omega_vel > self.omega_vel_max:
            omega_vel = self.omega_vel_max
        elif omega_vel < self.omega_vel_min:
            omega_vel = self.omega_vel_min

        return v_vel, omega_vel
        

    def codeB_prt8_state_transition(self, position, velocity):
        # unpack
        actual_x = position[0]
        actual_y = position[1]
        actual_theta = position[2]
        v_vel = velocity[0]
        omega_vel = velocity[1]

        # calculate next configuration
        actual_x += np.cos(actual_theta) * (v_vel * self.dt)
        actual_y += np.sin(actual_theta) * (v_vel * self.dt)
        actual_theta += omega_vel * self.dt

        # add noise
        actual_x += (0.01) * np.random.random() * np.random.choice([-1, 1])
        actual_y += (0.01) * np.random.random() * np.random.choice([-1, 1])
        actual_theta += 0.05 * np.random.random() * np.random.choice([-1, 1])

        return actual_x, actual_y, actual_theta
    
    def codeB_prt8_simulator(self):
        # initialize starting position and velocities
        self.x = self.start[0]
        self.y = self.start[1]
        self.theta = -np.pi/2
        self.v_vel = 0.0
        self.omega_vel = 0.0

        self.config_hist = [[self.x, self.y, self.theta]]
        self.vel_hist = [[self.v_vel, self.omega_vel]]
        self.goal_hist = []

        self.plan_crdnt = self.plan

        for cell in self.plan_crdnt:
            goal = self.idx_to_crdnt(cell) 
            
            # set goal as center of cell
            goal[0] += self.scale/2
            goal[1] += self.scale/2

            # set initial tolerance and distance_error
            tolerance =  self.scale * 0.25
            distance_error = 10000          

            # check if neighbors are occupied
            neighbors, occupied = self.find_if_cardinal_neighbors_occupied(cell)
            for ii in range(len(occupied)):
                if occupied[ii] == 1:
                    dif = np.array((neighbors[ii])) - np.array((cell))
                    goal += -dif * self.scale * 0.75

            # make sure it gets to the goal coordinate
            if (np.array(goal) == np.array(self.plan_crdnt[-1])).all(): 
                goal = self.goal                # make sure it goes to goal coordinates
                tolerance = 0.01     # tight tolerance to ensure it reaches goal state

            while(distance_error > tolerance):  
                 # run controller, get new velocities
                self.v_vel, self.omega_vel = self.codeB_prt8_controller(goal, [self.x, self.y, self.theta], [self.v_vel, self.omega_vel])

                # estimate new position (with noise)
                self.x, self.y, self.theta = self.codeB_prt8_state_transition([self.x, self.y, self.theta], [self.v_vel, self.omega_vel])

                # calculate distance from goal
                distance_error = np.sqrt((goal[0] - self.x)**2 + (goal[1] - self.y)**2)

                # save history
                self.config_hist.append([self.x, self.y, self.theta])
                self.vel_hist.append([self.v_vel, self.omega_vel])
            
            self.goal_hist.append(goal)

        if self.prt:    
            print(f'config_hist {np.shape(self.config_hist)}\n')
            print(f'vel_hist {np.shape(self.vel_hist)}\n')
            print(f'goal_hist {np.shape(self.goal_hist)}\n')
        
    def visualize_path_and_simulation(self,name,save):
        """
        Visualize grid with obstacles, path, and start and finish cells marked.
        Add the simulated robot path on top. 
        """
        # convert hist arrays to numpy
        config_hist = np.array(self.config_hist)
        vel_hist = np.array(self.vel_hist)
        goal_hist = np.array(self.goal_hist)

        # create points for plotting edges
        lin_x = np.linspace(self.x_min,self.x_max,10)
        lin_y = np.linspace(self.y_min,self.y_max,10)

        # create figure
        fig, ax = plt.subplots(figsize=[7,12])
        
        # draw lines
        for r in self.row_label:
            x = r * np.ones([10])
            ax.plot(x,lin_y,'-k',linewidth=0.5)

        for c in self.col_label:
            y = c * np.ones([10])
            ax.plot(lin_x,y,'-k',linewidth=0.5)

        # draw occupied cells
        for cell in self.occupied_cells:
            ################### https://www.geeksforgeeks.org/python/how-to-draw-shapes-in-matplotlib-with-python/ ############
            rect = Rectangle([self.row_label[cell[0]],self.col_label[cell[1]]],self.scale,self.scale,edgecolor='black',facecolor='black')
            ax.add_patch(rect)
        # again once for the legend label
        rect = Rectangle([self.row_label[cell[0]],self.col_label[cell[1]]],self.scale,self.scale,edgecolor='black',facecolor='black',label='obstacle')
        ax.add_patch(rect)

        # draw path cells
        for cell in self.plan:
            rect = Rectangle([self.row_label[cell[0]],self.col_label[cell[1]]],self.scale,self.scale,edgecolor='red',facecolor='red')
            ax.add_patch(rect)
        # again once for the legend label
        rect = Rectangle([self.row_label[cell[0]],self.col_label[cell[1]]],self.scale,self.scale,edgecolor='red',facecolor='red',label='path')
        ax.add_patch(rect)

        # draw goal cell 
        rect = Rectangle(self.idx_to_crdnt(self.goal_index),self.scale,self.scale,edgecolor='yellow',facecolor='yellow',label='goal')
        ax.add_patch(rect)

        # draw start cell
        rect = Rectangle(self.idx_to_crdnt(self.start_index),self.scale,self.scale,edgecolor='green',facecolor='green',label='start')
        ax.add_patch(rect)

        # draw the robot path
        ax.plot(config_hist[:,0], config_hist[:,1],'m-',linewidth=2.0,label='sim_robot')
        
        # draw the robot headings along its path
        if self.scale == 1.0:
            decimation_factor = int(len(config_hist)/50)        
        else:
            decimation_factor = int(len(config_hist)/100)        
        for ii in range(0,np.shape(config_hist)[0],decimation_factor):  # calculate  direction
            dx = .1 * np.cos(config_hist[ii,2])
            dy = .1 * np.sin(config_hist[ii,2])
            if self.scale == 1.0:
                plt.arrow(config_hist[ii,0], config_hist[ii,1], dx, dy, head_width=0.1, head_length=0.25, length_includes_head = 0, fc='magenta', ec='black')
            else:
                plt.arrow(config_hist[ii,0], config_hist[ii,1], dx, dy, head_width=0.035, head_length=0.05, length_includes_head = 1, fc='magenta', ec='black')

        
        # fit graph to area of path
        xlim = np.array([min(config_hist[:,0])-self.scale*2,max(config_hist[:,0])+self.scale*2])
        ylim = np.array([min(config_hist[:,1])-self.scale*2,max(config_hist[:,1])+self.scale*2])
        w = (xlim[1]-xlim[0])
        l = (ylim[1]-ylim[0])
        ratio = w/l
        desired_ratio = 5/7
        if np.isclose(ratio,desired_ratio,atol=0.25):
            pass
        elif ratio > desired_ratio:
            new_length = ratio/(desired_ratio) * l
            dif = new_length - l
            ylim[0] += dif
            ylim[1] -= dif
            ylim = [min(ylim),max(ylim)]
        elif ratio < desired_ratio:
            new_width = ratio/(desired_ratio) * w
            dif = new_width - w
            xlim[0] += dif
            xlim[1] -= dif

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(name)

        ax.legend()
        fig.set_size_inches(5,7)

        # plt.savefig(save, dpi=1000)

        plt.show()

    def codeB_part10_Online(self,start,goal,run,part):
        """
        Online A* algorithm implementation

        Produces a path from start to goal

        Also simulates a robot moving between the planned points

        PARAMETERS:
        start (list): x,y coordinates of start position
        goal (list): x,y coordinates of goal position

        OUTPUTS:
        self.path (list): list of cells along path from start to finish including start and goal cells
        """
        # save start and end position, convert to row col indexes
        self.start = start
        self.goal = goal
        self.start_index = self.crdnt_to_idx(start)
        self.goal_index = self.crdnt_to_idx(goal)
        goal_crdnt_center = np.array(self.idx_to_crdnt(self.goal_index)) + self.scale/2

        # initialize robot sim variables
        self.x = start[0]
        self.y = start[1]
        self.theta = -np.pi/2
        self.v_vel = 0.0
        self.omega_vel = 0.0

        # initialize history arrays
        self.config_hist = [[self.x, self.y, self.theta]]
        self.vel_hist = [[self.v_vel, self.omega_vel]]
        self.goal_hist = []

        # if printing is enabled, print
        if self.prt:
            print(f'start {np.shape(start)}:\n{start}\n')
            print(f'goal {np.shape(goal)}:\n{goal}\n')
            print(f'start_index {np.shape(self.start_index)}:\n{self.start_index}\n')
            print(f'goal_index {np.shape(self.goal_index)}:\n{self.goal_index}\n')

        # instantiate open set with start row col index as part of open set
        open_set = [self.start_index]

        # initialize total cost array and list for nodes in path
        f_list = []
        path_list = []

        # loop
        while(1):
            # loop through open set
            for cell in open_set:
                # calculate heuristic (distance function in grid plane)
                crdnt = self.idx_to_crdnt(cell)
                crdnt = np.array(crdnt) + self.scale/2
                # h = np.sqrt(np.square(cell[0]-self.goal_index[0])+np.square(cell[1]-self.goal_index[1])) # this one works better
                h = np.sqrt(np.square(crdnt[0]-goal_crdnt_center[0])+np.square(crdnt[1]-goal_crdnt_center[1])) # this one works better
                # true cost, provided by assignment
                if cell in self.occupied_cells:
                    g = 1000  # if it is an occupied cell
                else:
                    g = 1

                # calculate total cost and save
                f_list.append(g + h)
            
            # find lowest cost point
            min_idx = np.argmin(f_list)

            # clear f_list
            f_list = []

            # check if goal is reached
            if open_set[min_idx][0] == self.goal_index[0] and open_set[min_idx][1] == self.goal_index[1]:
                path_list.append(open_set.pop(min_idx))

                # run robot controller in real time
                self.codeB_prt10_simulator(path_list[-1],last=True)

                # visualize at every step if you would like (optional)
                # self.visualize_path_and_simulation(name=f"Real time planning and execution {len(path_list)}",save=f"CodeB_prt{part}_plots/run-{run}_img-{len(path_list)}")
                break

            # get neighbors of lowest cost cell
            n = self.get_neighbors(open_set[min_idx])

            # remove lowest cost cell from open set
            path_list.append(open_set.pop(min_idx))
            self.plan = path_list

            # add neighbors to open set, if they are not in already
            for p in n:
                # check and make sure it is not already in the open set or that it is in the path
                if p not in open_set and p not in path_list:
                    # add to open set
                    open_set.append(p)

            # run robot controller in real time
            self.codeB_prt10_simulator(path_list[-1],last=False)

            # visualize at every step if you would like (optional)
            # self.visualize_path_and_simulation(name=f"Real time planning and execution {len(path_list)}",save=f"CodeB_prt{part}_plots/run-{run}_img-{len(path_list)}")


        # save path to object
        self.plan = path_list

    def codeB_prt10_controller(self, goal, position, velocity):
        # unpack parameters
        goal_x = goal[0]
        goal_y = goal[1]
        actual_x = position[0]
        actual_y = position[1]
        actual_theta = position[2]
        v_vel = velocity[0]
        omega_vel = velocity[1]

        distance_error = np.sqrt((goal_x - actual_x)**2 + (goal_y - actual_y)**2)
        heading_error = np.atan2((goal_y - actual_y),(goal_x - actual_x)) - actual_theta

        if abs(heading_error) > 3.14:
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        # controller
        if abs(heading_error) < 0.1:
            ideal_v_vel = State.FORWARD
        else:
            ideal_v_vel = State.STOPPED


        if heading_error > 0.0 and abs(heading_error) > 0.1:
            ideal_omega_vel = State.LEFT
        else:
            ideal_omega_vel = State.RIGHT
        
        if ideal_v_vel == State.FORWARD:
            v_vel += self.v_dot_max * self.dt
        elif ideal_v_vel == State.STOPPED:
            v_vel += -self.v_dot_max * self.dt
        
        if ideal_omega_vel == State.LEFT:
            omega_vel += self.omega_dot_max * self.dt
        elif ideal_omega_vel == State.RIGHT:
            omega_vel += -self.omega_dot_max * self.dt

        # clamping
        if v_vel > self.v_vel_max:
            v_vel = self.v_vel_max
        elif v_vel < self.v_vel_min:
            v_vel = self.v_vel_min

        if omega_vel > self.omega_vel_max:
            omega_vel = self.omega_vel_max
        elif omega_vel < self.omega_vel_min:
            omega_vel = self.omega_vel_min

        return v_vel, omega_vel

    def codeB_prt10_state_transition(self, position, velocity):
        # unpack
        actual_x = position[0]
        actual_y = position[1]
        actual_theta = position[2]
        v_vel = velocity[0]
        omega_vel = velocity[1]

        # calculate next configuration
        actual_x += np.cos(actual_theta) * (v_vel * self.dt)
        actual_y += np.sin(actual_theta) * (v_vel * self.dt)
        actual_theta += omega_vel * self.dt

        # add noise
        actual_x += (0.1 * self.scale) * np.random.random() * np.random.choice([-1, 1])
        actual_y += (0.1 * self.scale) * np.random.random() * np.random.choice([-1, 1])
        actual_theta += 0.1 * np.random.random() * np.random.choice([-1, 1])

        return actual_x, actual_y, actual_theta
    
    def codeB_prt10_simulator(self, cell, last):
        # transform cell to coordinates, then center it in the cell
        goal = self.idx_to_crdnt(cell) 
            
        # set goal as center of cell
        goal[0] += self.scale/2
        goal[1] += self.scale/2

        # set initial tolerance and distance_error
        tolerance =  self.scale * 0.1
        distance_error = 10000          

        # check if neighbors are occupied
        neighbors, occupied = self.find_if_cardinal_neighbors_occupied(cell)
        for ii in range(len(occupied)):
            if occupied[ii] == 1:
                dif = np.array((neighbors[ii])) - np.array((cell))
                goal += -dif * self.scale * 0.75
                
        # make sure it gets to the goal coordinate
        if last == True: 
            goal = self.goal                # make sure it goes to goal coordinates
            tolerance = 0.01     # tight tolerance to ensure it reaches goal state

        while(distance_error > tolerance):  
                # run controller, get new velocities
            self.v_vel, self.omega_vel = self.codeB_prt8_controller(goal, [self.x, self.y, self.theta], [self.v_vel, self.omega_vel])

            # estimate new position (with noise)
            self.x, self.y, self.theta = self.codeB_prt8_state_transition([self.x, self.y, self.theta], [self.v_vel, self.omega_vel])

            # calculate distance from goal
            distance_error = np.sqrt((goal[0] - self.x)**2 + (goal[1] - self.y)**2)

            # save history
            self.config_hist.append([self.x, self.y, self.theta])
            self.vel_hist.append([self.v_vel, self.omega_vel])
        
            self.goal_hist.append(goal)

        if self.prt:    
            print(f'config_hist {np.shape(self.config_hist)}\n')
            print(f'vel_hist {np.shape(self.vel_hist)}\n')
            print(f'goal_hist {np.shape(self.goal_hist)}\n')
def main():
    g1 = Grid([-2,5],[-6,6],1)
    g1.visualize_grid(name='Empty Grid') 

    g1.codeA_part2_Offline(start=[0.5,-1.5],goal=[0.5,1.5])
    g1.visualize_path(name='Code A part 3 plot_1')

    g1.codeA_part2_Offline(start=[4.5,3.5],goal=[4.5,-1.5])
    g1.visualize_path(name='Code A part 3 plot_2')

    g1.codeA_part2_Offline(start=[-0.5,5.5],goal=[1.5,-3.5])
    g1.visualize_path(name='Code A part 3 plot_3')

    g1.codeA_part4_Online(start=[0.5,-1.5],goal=[0.5,1.5])
    g1.visualize_path(name='Code A part 5 plot_1')

    g1.codeA_part4_Online(start=[4.5,3.5],goal=[4.5,-1.5])
    g1.visualize_path(name='Code A part 5 plot_2')

    g1.codeA_part4_Online(start=[-0.5,5.5],goal=[1.5,-3.5])
    g1.visualize_path(name='Code A part 5 plot_3')

    g2 = Grid([-2,5],[-6,6],.1)
    g2.visualize_grid(name='Empty, higher res Grid') 

    g2.codeA_part4_Online(start=[2.45,-3.55],goal=[0.95,-1.55])
    g2.visualize_path(name='Code A part 7 plot_1')

    g2.codeA_part4_Online(start=[4.95,-0.05],goal=[2.45, 0.25])
    g2.visualize_path(name='Code A part 7 plot_2')

    g2.codeA_part4_Online(start=[0.55, 1.45],goal=[1.95, 3.95])
    g2.visualize_path(name='Code A part 7 plot_3')

    g3 = Grid([-2,5],[-6,6],.1)
    g3.codeA_part4_Online(start=[2.45,-3.55],goal=[0.95,-1.55])
    g3.codeB_prt8_simulator()
    g3.visualize_path_and_simulation(name='Code B part 9 plot_1',save=f"CodeB_prt9_plot_1")
    g3.codeA_part4_Online(start=[4.95,-0.05],goal=[2.45, 0.25])
    g3.codeB_prt8_simulator()
    g3.visualize_path_and_simulation(name='Code B part 9 plot_2',save=f"CodeB_prt9_plot_2")
    g3.codeA_part4_Online(start=[0.55, 1.45],goal=[1.95, 3.95])
    g3.codeB_prt8_simulator()
    g3.visualize_path_and_simulation(name='Code B part 9 plot_3',save=f"CodeB_prt9_plot_3")

    g4 = Grid([-2,5],[-6,6],.1)
    g4.codeB_part10_Online(start=[2.45,-3.55],goal=[0.95,-1.55],run=1,part=10)
    g4.visualize_path_and_simulation(name='Code B part 10 plot_1',save=f"CodeB_prt10_plot_1")
    g4.codeB_part10_Online(start=[4.95,-0.05],goal=[2.45, 0.25],run=2,part=10)
    g4.visualize_path_and_simulation(name='Code B part 10 plot_2',save=f"CodeB_prt10_plot_2")
    g4.codeB_part10_Online(start=[0.55, 1.45],goal=[1.95, 3.95],run=3,part=10)
    g4.visualize_path_and_simulation(name='Code B part 10 plot_3',save=f"CodeB_prt10_plot_3")

    g5 = Grid([-2,5],[-6,6],.1)
    g5.codeB_part10_Online(start=[0.5,-1.5],goal=[0.5,1.5],run=1,part=11.1)
    g5.visualize_path_and_simulation(name='Code B part 11-1 plot_1',save=f"CodeB_prt11-1_plot_1")
    g5.codeB_part10_Online(start=[4.5,3.5],goal=[4.5,-1.5],run=2,part=11.1)
    g5.visualize_path_and_simulation(name='Code B part 11-1 plot_2',save=f"CodeB_prt11-1_plot_2")
    g5.codeB_part10_Online(start=[-0.5,5.5],goal=[1.5,-3.5],run=3,part=11.1)
    g5.visualize_path_and_simulation(name='Code B part 11-1 plot_3',save=f"CodeB_prt11-1_plot_3")

    g6 = Grid([-2,5],[-6,6],1)
    g6.codeB_part10_Online(start=[0.5,-1.5],goal=[0.5,1.5],run=1,part=11.2)
    g6.visualize_path_and_simulation(name='Code B part 11-2 plot_1',save=f"CodeB_prt11-2_plot_1")
    g6.codeB_part10_Online(start=[4.5,3.5],goal=[4.5,-1.5],run=2,part=11.2)
    g6.visualize_path_and_simulation(name='Code B part 11-2 plot_2',save=f"CodeB_prt11-2_plot_2")
    g6.codeB_part10_Online(start=[-0.5,5.5],goal=[1.5,-3.5],run=3,part=11.2)
    g6.visualize_path_and_simulation(name='Code B part 11-2 plot_3',save=f"CodeB_prt11-2_plot_3")

if __name__ == '__main__':
    main()
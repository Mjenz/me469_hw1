import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

        # this does not work for all sclaes!!!! use grid size?
        x_inc = 0 if (index[0] + 1) >=  np.shape(self.grid)[0] else 1
        x_dec = 0 if (index[0] - 1) <   0 else 1
        y_inc = 0 if (index[1] + 1) >=  np.shape(self.grid)[1] else 1
        y_dec = 0 if (index[1] - 1) <   0 else 1

        print(x_inc,x_dec,y_inc,y_dec)
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
    
    def codeA_part2_Astar(self,start,goal):
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
                h = np.sqrt(np.square(cell[0]-self.goal_index[0])+np.square(cell[1]-self.goal_index[1])) # this one works better
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

    def visualize_grid(self):
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
        ax.set_title('Occupancy Grid Visualization')
        fig.tight_layout()
        fig.show()    

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

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('Occupancy Grid Visualization')
        ax.legend()
        plt.savefig(name)
        plt.show()


def main():
    g1 = Grid([-2,5],[-6,6],1)
    # g1.visualize_grid() 

    g1.codeA_part2_Astar(start=[0.5,-1.5],goal=[0.5,1.5])
    g1.visualize_path(name='plot_1')

    g1.codeA_part2_Astar(start=[4.5,3.5],goal=[4.5,-1.5])
    g1.visualize_path(name='plot_2')

    g1.codeA_part2_Astar(start=[-0.5,5.5],goal=[1.5,-3.5])
    g1.visualize_path(name='plot_3')

    g2 = Grid([-2,5],[-6,6],.1)
    # g2.visualize_grid() 

    g2.codeA_part2_Astar(start=[2.45,-3.55],goal=[0.95,1.55])
    g2.visualize_path(name='plot_4')

    g2.codeA_part2_Astar(start=[4.95,0.05],goal=[2.45,0.25])
    g2.visualize_path(name='plot_5')

    g2.codeA_part2_Astar(start=[-0.55,1.45],goal=[1.95,-3.95])
    g2.visualize_path(name='plot_6')
    
if __name__ == '__main__':
    main()
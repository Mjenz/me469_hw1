import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Grid():
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



        obstacles = np.loadtxt('ds1_Landmark_Groundtruth.dat')

        loc_hist = []
        self.occupied_cells = []
        occupied_cells_readable = []

        for obs in obstacles:
            x = obs[1]
            y = obs[2]
            loc_hist.append([float(round(x,2)),float(round(y,2))])
            for ii in range(len(self.row_label)-1):
                if x >= self.row_label[ii] and x < self.row_label[ii+1]:
                    for jj in range(len(self.col_label)-1):
                        if y >= self.col_label[jj] and y < self.col_label[jj+1]:
                            occupied_cells_readable.append([self.row_label[ii],self.col_label[jj]]) # cols, rows (x, y)
                            self.occupied_cells.append([ii,jj]) # rows, cols (y, x)

        # expand object markers if using smaller cell size
        if self.scale == 0.1:
            initial_occupied = self.occupied_cells.copy()

            for cell in initial_occupied:
                ii  = cell[0]
                jj  = cell[1]
                for i in range(ii-3,ii+3):
                    for j in range(jj-3,jj+3):
                        if ii == i and jj == j:
                            pass
                        else:
                            self.occupied_cells.append([i,j]) # rows, cols (y, x)

        for cell in self.occupied_cells:
            self.grid[cell[0],cell[1]] = 1   

        if self.prt:
            print(f'grid {np.shape(self.grid)}:\n{self.grid}\n')
            print(f'row_label {np.shape(self.row_label)}:\n{self.row_label}\n')
            print(f'col_label {np.shape(self.col_label)}:\n{self.col_label}\n')
            print(f'loc_hist {np.shape(loc_hist)}:\n{loc_hist}\n')
            print(f'occupied_cells_readable {np.shape(occupied_cells_readable)}:\n{occupied_cells_readable}\n')
            print(f'occupied_cells {np.shape(self.occupied_cells)}:\n{self.occupied_cells}\n')
            print(f'grid {np.shape(self.grid)}:\n{self.grid}\n')

    def crdnt_to_idx(self,coordinate):
        for ii in range(len(self.row_label)-1):
            if coordinate[0] >= self.row_label[ii] and coordinate[0] < self.row_label[ii+1]:
                for jj in range(len(self.col_label)-1):
                    if coordinate[1] >= self.col_label[jj] and coordinate[1] < self.col_label[jj+1]:
                        return [ii,jj]

    def idx_to_crdnt(self,index):
        return [index[0] * self.scale + self.x_min, index[1] * self.scale + self.y_min]
    
    def get_neighbors(self,index):
        n = []
        print(self.x_max,self.x_min,self.y_max,self.y_min)

        # this does not work for all sclaes!!!!
        x_inc = 0 if (index[0] + 1) >  (self.x_max + 1) else 1
        x_dec = 0 if (index[0] + 1) <  (self.x_min + 1) else 1
        y_inc = 0 if (index[1] + 1) >  (self.y_max + 1) else 1
        y_dec = 0 if (index[1] + 1) <  (self.y_min + 1) else 1

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
        self.start = start
        self.goal = goal
        self.start_index = self.crdnt_to_idx(start)
        self.goal_index = self.crdnt_to_idx(goal)

        print(f'start {np.shape(start)}:\n{start}\n')
        print(f'goal {np.shape(goal)}:\n{goal}\n')
        print(f'start_index {np.shape(self.start_index)}:\n{self.start_index}\n')
        print(f'goal_index {np.shape(self.goal_index)}:\n{self.goal_index}\n')


        open_set = [self.start_index]
        f_list = []
        path_list = []

        count = 0
        while(1):
            for cell in open_set:
                # heuristic
                h = np.sqrt(np.square(cell[0]-self.goal_index[0])+np.square(cell[1]-self.goal_index[1])) # this one works better
                # true cost, provided by instructor
                if cell in self.occupied_cells:
                    g = 1000
                else:
                    g = 1
                # total cost
                f = g + h
                # save
                f_list.append(f)
            
            # find lowest cost point
            min_idx = np.argmin(f_list)
            print(open_set[min_idx],f_list[min_idx])
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
            # visualize at every step if you would like
            self.visualize_path()
        # save path to object
        self.plan = path_list

    def visualize_grid(self):
        lin_x = np.linspace(self.x_min,self.x_max,10)
        lin_y = np.linspace(self.y_min,self.y_max,10)

        fig, ax = plt.subplots(figsize=[7,12])
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

        plt.show()

    
    def visualize_path(self):
        lin_x = np.linspace(self.x_min,self.x_max,10)
        lin_y = np.linspace(self.y_min,self.y_max,10)

        fig, ax = plt.subplots(figsize=[7,12])
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

        for cell in self.plan:
            rect = Rectangle([self.row_label[cell[0]],self.col_label[cell[1]]],self.scale,self.scale,edgecolor='red',facecolor='red')
            ax.add_patch(rect)

        rect = Rectangle(self.idx_to_crdnt(self.goal_index),self.scale,self.scale,edgecolor='yellow',facecolor='yellow')
        ax.add_patch(rect)

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('Occupancy Grid Visualization')

        plt.show()


def main():
    g = Grid([-2,5],[-6,6],.1)
    # g.visualize_grid() 

    # g.codeA_part2_Astar(start=[0.5,-1.5],goal=[0.5,1.5])
    # g.visualize_path()

    g.codeA_part2_Astar(start=[4.5,3.5],goal=[4.5,-1.5])
    g.visualize_path()

    # g.codeA_part2_Astar(start=[-0.5,5.5],goal=[1.5,-3.5])
    # g.visualize_path()
if __name__ == '__main__':
    main()
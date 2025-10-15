import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Grid():
    def __init__(self,x_range,y_range,scale):
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
        
        # intialize the grid of zeros
        self.grid = np.zeros([(x_range[1]-x_range[0]) * int(1/self.scale),((y_range[1]-y_range[0]) * int(1/self.scale))])

        # initialize labels for each axis. columns correspond to y and rows to x for ease of use purposes
        self.col_label = [y_range[0] + ii * self.scale for ii in range(np.shape(self.grid)[1]+1)]
        self.row_label = [x_range[0] + jj * self.scale for jj in range(np.shape(self.grid)[0]+1)]

        print(f'grid {np.shape(self.grid)}:\n{self.grid}\n')
        print(f'row_label {np.shape(self.row_label)}:\n{self.row_label}\n')
        print(f'col_label {np.shape(self.col_label)}:\n{self.col_label}\n')

        obstacles = np.loadtxt('ds1_Landmark_Groundtruth.dat')

        loc_hist = []
        self.occupied_cells = []
        occupied_cells_readable = []

        for obs in obstacles:
            x = obs[1]
            y = obs[2]
            loc_hist.append([float(round(x,2)),float(round(y,2))])
            for ii in range(len(self.row_label)-1):
                if x > self.row_label[ii] and x < self.row_label[ii+1]:
                    for jj in range(len(self.col_label)-1):
                        if y > self.col_label[jj] and y < self.col_label[jj+1]:
                            # print(self.row_label[ii],self.col_label[jj])

                            occupied_cells_readable.append([self.row_label[ii],self.col_label[jj]]) # cols, rows (x, y)
                            self.occupied_cells.append([ii,jj]) # rows, cols (y, x)

        for cell in self.occupied_cells:
            self.grid[cell[0],cell[1]] = 1   

        print(f'loc_hist {np.shape(loc_hist)}:\n{loc_hist}\n')
        print(f'occupied_cells_readable {np.shape(occupied_cells_readable)}:\n{occupied_cells_readable}\n')
        print(f'occupied_cells {np.shape(self.occupied_cells)}:\n{self.occupied_cells}\n')
        print(f'grid {np.shape(self.grid)}:\n{self.grid}\n')

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

    def A_start_CodeA_prt2(self):
        pass # help me
def main():
    g = Grid([-2,5],[-6,6],.1)
    g.visualize_grid()
if __name__ == '__main__':
    main()
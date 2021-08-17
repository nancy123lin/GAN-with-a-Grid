import numpy as np



class FloorPlan:
    def __init__(self, length, width, height=None, scale='ft'):
        # scale defaults to 'ft', but could also be 'm', however it's always 'ft' operationally
        if scale == 'ft':
            self.length = length
            self.width = width
            self.height = height
        elif scale == 'm':
            self.length = length * 3.28084
            self.width = width * 3.28084
            if self.height is None:
                self.height = None
            else:
                self.height = height * 3.28084
        self.grid = None

    def get_grid(self, mixture_size):
        # mixture_size is analogous to the 3 (r,b,g) for pictures, here it allows for complexity of the result
        grid_length = int(self.length // 1)
        if self.length - grid_length > 0.5:
            grid_length += 1
        grid_width = int(self.width // 1)
        if self.width - grid_width > 0.5:
            grid_width += 1
        if self.height is not None:
            grid_height = int(self.height // 1)
            if self.height - grid_height > 0.5:
                grid_height += 1
            self.grid = np.zeros((grid_length,grid_width,grid_height,mixture_size))
        else:
            self.grid = np.zeros((grid_length,grid_width,mixture_size))





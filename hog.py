import numpy as np
from tqdm import tqdm
import cv2

class HoG :

    def __init__ (self, orientations=8, pixels_per_cell=(8,8), cells_per_block=(2, 2)) :
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
    
    def gradients (self, img) :
        # Compute the gradients magnitude and orientation on an image of shape 32 x 32 (no color channel) 
        # Here simply apply Gx = [-1,0,1] and Gy = [-1,0,1]^T as it's simpler to code than convolutions with 2D kernels
        Gx, Gy = np.zeros(img.shape), np.zeros(img.shape)
        Gx[1:-1, :] = img[2:, :] - img[:-2, :]
        Gy[:, 1:-1] = img[:, 2:] - img[:, :-2]
        G = np.sqrt(Gx**2 + Gy**2)
        G_hat = (np.arctan(Gy/(Gx + 1e-10))*180/np.pi) % 180 # conversion in degree
        return G, G_hat

    def get_X(self, X, visualization=False):
        features = None

        for i in tqdm(range(len(X))):
            img = X[i]  # No need to reshape as images are already in the correct shape

            # Convert the image to the appropriate depth
            img = cv2.convertScaleAbs(img)

            # Step 2 - Gradients' magnitude and orientation computation:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert image to grayscale
            G, G_hat = self.gradients(gray_img)

            # Step 3 - Compute gradients histograms per cell :
            c_x, c_y = self.pixels_per_cell
            nb_cells_x = int(img.shape[0] // c_x)
            nb_cells_y = int(img.shape[1] // c_y)

            orientation_histogram = np.zeros((nb_cells_x, nb_cells_y, self.orientations), dtype=float)

            for o in range(self.orientations):
                o_low, o_sup = (180 / self.orientations) * o, (180 / self.orientations) * (o + 1)
                mask_o = (G_hat >= o_low) * (G_hat < o_sup)
                masked_G = G * mask_o

                for x in range(nb_cells_x):
                    for y in range(nb_cells_y):
                        orientation_histogram[x, y, o] = np.mean(
                            masked_G[x * c_x: (x + 1) * c_x, y * c_y: (y + 1) * c_y])

            if visualization:
                v = self.visualize_histograms(orientation_histogram)

            # Step 4 - Normalization per block of cells :
            b_x, b_y = self.cells_per_block
            nb_blocks_x = (nb_cells_x - b_x) + 1
            nb_blocks_y = (nb_cells_y - b_y) + 1
            normalized_blocks = np.zeros((nb_blocks_x, nb_blocks_y, b_x, b_y, self.orientations), dtype=float)

            for x in range(nb_blocks_x):
                for y in range(nb_blocks_y):
                    block = orientation_histogram[x: x + b_x, y: y + b_y, :]
                    normalized_blocks[x, y, :] = block / np.sqrt(np.sum(block ** 2) + (1e-10) ** 2)

            # Step 5 - ravel to obtain a features vector
            f = normalized_blocks.ravel()

            # Add our features vector to the other ones :
            if features is None:
                nb_features = len(f)
                features = np.zeros((len(X), nb_features))
            features[i, :] = f

        return features


    def visualize_histograms (self, orientation_histogram) :
            visu = np.zeros((32,32), dtype=float)

            c_x, c_y = self.pixels_per_cell
            nb_cells_x, nb_cells_y, _ = orientation_histogram.shape

            # Prepare the midpoints of each cell to be abble to drax lines with some orientations
            orientations_array = np.arange(self.orientations)
            orientation_midpoints_rad = np.pi * (orientations_array + .5) / self.orientations # converted in rad

            radius = min(c_x, c_y) // 2 - 1
            orientation_midpoints_x = radius * np.sin(orientation_midpoints_rad)
            orientation_midpoints_y = radius * np.cos(orientation_midpoints_rad)

            # Fill the visualization per cell :
            for x in range(nb_cells_x) :
                for y in range(nb_cells_y) : 
                    # we are in cell (x,y), which grandients' histo is in orientation_histogram[x,y,:]
                    center_cell = x * c_x + (c_x // 2) ,  y * c_y + (c_y // 2)
                    for o in orientations_array : 
                        # For each orientation, we draw a line with "magnitude" given in orientation_histogram[x,y,o]
                        line_x, line_y = line(int(center_cell[0] - orientation_midpoints_x[o]),
                                       int(center_cell[1] + orientation_midpoints_y[o]),
                                       int(center_cell[0] + orientation_midpoints_x[o]),
                                       int(center_cell[1] - orientation_midpoints_y[o]))
                        visu[line_x, line_y] += orientation_histogram[x, y, o]
            
            # Plot :
            plt.imshow(visu)
            plt.show()
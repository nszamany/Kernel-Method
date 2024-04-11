
import cv2

class myTransform:
    
    def flip_image_horizontal(self, image):
        # Takes an image as input and outputs the same image with a horizontal flip
        result = image.copy()
        for channel in range(3):
            aux = image[:, :, channel]
            for column in range(len(aux)):
                result[:, column, channel] = aux[:, len(aux) - column - 1]
        return result
    
    def gaussian_blurr(self, image):
        # Takes an image as input and outputs the same image with a blur
        result = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)      
        return result
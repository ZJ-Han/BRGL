import cv2
import os


folder_path = r''


for filename in os.listdir(folder_path):
    if filename.endswith('.jpeg') or filename.endswith('.png'):

        image_path = os.path.join(folder_path, filename)


        image = cv2.imread(image_path, 0) 

        equalized_image = cv2.equalizeHist(image)


        equalized_filename = 'equalized_' + filename
        equalized_image_path = os.path.join(folder_path, equalized_filename)
        cv2.imwrite(equalized_image_path, equalized_image)


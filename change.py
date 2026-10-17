from PIL import Image
import os

input_folder = r'D:\desktop\newmutual\data\unlabel1\02'  
output_folder = r'D:\desktop\newmutual\data\test\02'  

tif_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

for tif_file in tif_files:
    tif_path = os.path.join(input_folder, tif_file)
    img = Image.open(tif_path)



    jpeg_path = os.path.join(output_folder, os.path.splitext(tif_file)[0] + '.jpeg')
    img.save(jpeg_path, 'JPEG')

print('done！')
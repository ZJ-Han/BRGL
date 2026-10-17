from PIL import Image
import os


def convert_tif_to_jpeg(input_folder, output_folder):

    for root, _, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(".tif"):
                input_path = os.path.join(root, filename)
                relative_path = os.path.relpath(input_path, input_folder)
                output_path = os.path.join(output_folder, os.path.splitext(relative_path)[0] + ".jpeg")

                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                image = Image.open(input_path)
                image.save(output_path, "JPEG")
                os.remove(input_path)


input_folder = r'D:\desktop\newmutual\data\labeldata2'
output_folder = r'D:\desktop\newmutual\data\labeldata2'

convert_tif_to_jpeg(input_folder, output_folder)
print("转换完成！")
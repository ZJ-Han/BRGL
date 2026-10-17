import os
import shutil

def getback(source_folder,tar_folder):

    for subfolder_name in os.listdir(source_folder):
        subfolder_path = os.path.join(source_folder, subfolder_name)

        if os.path.isdir(subfolder_path):

            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, filename)

                last_two_digits = filename[-7:-5] 

                destination_subfolder = os.path.join(tar_folder, last_two_digits)
                destination_file_path = os.path.join(destination_subfolder, filename)
  
                shutil.copy2(file_path, destination_file_path)
                os.remove(file_path)


def getbacklabel(source_folder,tar_folder):
    for subfolder_name in os.listdir(source_folder):
        subfolder_path = os.path.join(source_folder, subfolder_name)

        if os.path.isdir(subfolder_path):

            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, filename)

                destination_subfolder = os.path.join(tar_folder, subfolder_name )
                destination_file_path = os.path.join(destination_subfolder, filename)

                shutil.copy2(file_path, destination_file_path)
                os.remove(file_path)
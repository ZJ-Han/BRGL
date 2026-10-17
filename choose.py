
import os, random, shutil
def copyFile(fileDir, tarDir,num):
    pathDir = os.listdir(fileDir)
    sample = random.sample(pathDir, num)
    for name in sample:
        shutil.copyfile(fileDir + name, tarDir + name)
        os.remove(fileDir + name)
def copysample(fileDir, tarDir):
    #pathDir = os.listdir(fileDir)
   # sample = random.sample(pathDir, 5)
    #for name in pathDir:
        #shutil.copyfile(fileDir + name, tarDir + name)
    shutil.copytree(fileDir , tarDir )
      #  os.remove(fileDir + name)
def choose(classnum,num):
    rootdir1 = r'./data/unlabel1'
    rootdir2 = r'./data/labeldata1'
    rootdir3 = r'./data/unlabel2'
    rootdir4 = r'./data/labeldata2'
    shutil.rmtree(rootdir3)
    shutil.rmtree(rootdir4)

    for i in range(int(classnum)):
        fileDir = rootdir1 + os.sep + f"{i:02d}" + os.sep
        tarDir =  rootdir2 + os.sep+ f"{i:02d}" + os.sep
        if not os.path.exists(tarDir):
            os.makedirs(tarDir)
        copyFile(fileDir, tarDir,num)
    shutil.copytree(rootdir1, rootdir3)
    shutil.copytree(rootdir2, rootdir4)





    

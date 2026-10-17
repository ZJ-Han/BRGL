import os
import subprocess


for root, dirs, files in os.walk("/home/MSTAR/data/MSTAR-PublicClutter-CD1/MSTAR_PUBLIC_CLUTTER_CD1/CLUTTER/15_DEG/"):
    filename = files

a = "./mstar2jpeg -i /home/MSTAR/data/MSTAR-PublicClutter-CD1/MSTAR_PUBLIC_CLUTTER_CD1/CLUTTER/15_DEG/%s -o /home/data2/MSTAR-PublicClutter-CD1/MSTAR_PUBLIC_CLUTTER_CD1/CLUTTER/15_DEG/%s.jpeg -e qf"

for name in filename:
    b = a % (name,name)
    subprocess.call(b,shell=True)

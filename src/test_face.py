import os
import subprocess

#subprocess.call(["/bin/bash","-c","open /home/lfan/Dropbox/runCoAtt/darknet/darknet"])
filePath='/home/lfan/Dropbox/runCoAtt/darknet/mydata/test.txt'
os.system('cd /home/lfan/Dropbox/runCoAtt/darknet/')
with open(filePath,'r') as testfile:
    while True:
       lines=testfile.readline()
       if len(lines)==0:
           break
       os.system('./darknet detector test cfg/obj.data cfg/yolo-obj.cfg backup_922/yolo-obj_final.weights '+lines)

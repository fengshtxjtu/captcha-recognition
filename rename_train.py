# coding: utf-8
import os;
import csv

def rename():
    path = "/usr/code/NN/captcha-recognition-master/images/4-c-1-g/train";
    filelist = os.listdir(path)  # 该文件夹下所有的文件（包括文件夹）
#    print filelist
    csvfile = open('/usr/code/NN/captcha-recognition-master/images/4-c-1-g/train/lables.csv', 'wb')
    writer = csv.writer(csvfile)
    lables = []
    i=0
    for files in filelist:# 遍历所有文件
        Olddir = os.path.join(path, files);  # 原来的文件路径
#        print Olddir
        if os.path.isdir(Olddir):
            continue;
        filename = os.path.splitext(files)[0];  # 文件名
        filetype = os.path.splitext(files)[1];  # 文件扩展名
#        print filename,filetype

        lab = gen_lable(i,filename)
        print lab
        lables.append(lab)
        print lables


        Newdir = os.path.join(path, "/usr/code/NN/captcha-recognition-master/images/4-c-1-g/train/"+str(i)+filetype);  # 新的文件路径
        print Newdir
        i = i+1
        os.rename(Olddir, Newdir);  # 重命名

    writer.writerows(lables)
    csvfile.close()

def gen_lable(i,filename):
    filename = str(filename)
    return [i,filename]




if __name__=="__main__":
    rename()


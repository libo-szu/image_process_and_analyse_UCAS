import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def rgb1gray(f, method="NTSC"):
    if (method != "NTSC" and method != "average"):
        print("error:method is not exist!")
        return None
    r,g,b=f[:,:,0],f[:,:,1],f[:,:,2]
    if(method=="average"):
        gray_img=(r+g+b)//3
    else:
        gray_img = r*0.2989+g*0.5870+b*0.1140
    return gray_img
def test2_plot(img_path):
    img=cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
    gray_img_average=rgb1gray(img,"average")
    gray_img_NTSC=rgb1gray(img)
    if(not os.path.exists("./session2/")):
        os.mkdir("./session2/")

    plt.subplot(1,3,1)
    plt.title("original image")
    plt.imshow(img)
    plt.subplot(1,3,2)
    plt.title("average")
    plt.imshow(gray_img_average,cmap="gray")
    plt.subplot(1,3,3)
    plt.title("NTSC")
    plt.imshow(gray_img_NTSC,cmap="gray")
    plt.savefig("./session2/"+img_path.split(".")[0]+".jpg")
    plt.show()
def test_2():
    img1_path="lena512color.tiff"
    img2_path="mandril_color.tif"
    test2_plot(img1_path)
    test2_plot(img2_path)
if __name__=="__main__":
    test_2()
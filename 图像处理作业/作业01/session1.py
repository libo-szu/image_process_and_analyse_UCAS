import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def scanLine4e(f, I, loc):
    if(loc!="row" and loc!="col"):
        print("error:loc not is row or col")
        return None
    if(loc=="row"):
        return f[I,:]
    else:
        return f[:,I]
def one_image_test1(img_path):
    img=cv2.imread(img_path,0)
    w,h=img.shape
    I_row=w//2
    I_col=h//2

    if(w%2!=0):
        s_row=scanLine4e(img, I_row, "row")
        x_row = np.linspace(0, len(s_row), len(s_row))
        plt.title("row")
        plt.plot(x_row, s_row)

    else:
        s_row1 = scanLine4e(img, I_row, "row")
        s_row2 = scanLine4e(img, I_row+1, "row")

        x_row = np.linspace(0, len(s_row1), len(s_row1))

        plt.subplot(1, 2, 1)
        plt.title("row1")
        plt.plot(x_row, s_row1)

        plt.subplot(1, 2, 2)
        plt.title("row12")
        plt.plot(x_row, s_row2)
    if(not os.path.exists("./session1/")):
        os.mkdir("./session1/")
    plt.savefig("./session1/session1_row_"+img_path.split(".")[0]+".jpg")
    plt.show()

    if(h%2!=0):
        s_col=scanLine4e(img, I_row, "col")
        x_col = np.linspace(0, len(s_col), len(s_col))
        plt.title("col")
        plt.plot(x_col, s_col)

    else:
        s_col1 = scanLine4e(img, I_col, "col")
        s_col2 = scanLine4e(img, I_col+1, "col")

        x_col = np.linspace(0, len(s_col1), len(s_col1))

        plt.subplot(1, 2, 1)
        plt.title("col1")
        plt.plot(x_col, s_col1)

        plt.subplot(1, 2, 2)
        plt.title("col2")
        plt.plot(x_col, s_col2)
    plt.savefig("./session1/session1_col_"+img_path.split(".")[0]+".jpg")
    plt.show()


def test_1():
    img1_path="cameraman.tif"
    img2_path="einstein.tif"
    one_image_test1(img1_path)
    one_image_test1(img2_path)


if __name__=="__main__":
    test_1()
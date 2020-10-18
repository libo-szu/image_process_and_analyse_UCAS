import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from  session2 import rgb1gray

def flip180(arr):
    arr_180 = arr.reshape(arr.size)
    arr_180 = arr_180[::-1]
    arr_180 = arr_180.reshape(arr.shape)
    return arr_180
def padding_row(f,padding,i):
    f1 = f[i, :]
    for i in range(padding - 1):
        f1 = np.concatenate((f1, f[i, :]), 0)
    f1 = f1.reshape(padding, -1)
    return f1

def padding_col(f,padding,i):
    f3 = f[:, i]
    for i in range(padding - 1):
        f3 = np.concatenate((f3, f[:, i]), 0)
    f3 = f3.reshape(-1, padding)
    return f3
def padding_conor(conor,padding):
    conor_np=np.ones((padding,padding))*conor
    return conor_np

def twodConv(f, w,padding_method="zero"):
    w=flip180(w)
    row,col=f.shape
    k=w.shape[0]
    padding=(w.shape[0]-1)//2
    padding_img=np.zeros((row+padding*2,col+padding*2))
    padding_img[padding:row + padding, padding:col + padding] = f
    if(padding_method=="replicate"):
        # print(f.shape)
        f1=padding_row(f, padding, 0)
        f2=padding_row(f, padding, row-1)
        f3=padding_col(f,padding,0)
        f4=padding_col(f,padding,col-1)


        padding_img[:padding,padding:col+padding]=f1
        padding_img[row+padding:,padding:col+padding]=f2
        padding_img[padding:row + padding,:padding] = f3
        padding_img[padding:row + padding,col + padding:] =f4
        conor_1=padding_conor(f[0][0],padding)
        padding_img[:padding,:padding]=conor_1

        conor_2=padding_conor(f[row-1][0],padding)
        padding_img[padding+row:,:padding]=conor_2

        conor_3=padding_conor(f[0][col-1],padding)
        padding_img[:padding,padding+col:]=conor_3

        conor_4=padding_conor(f[row-1][col-1],padding)
        padding_img[row+padding:,padding+col:]=conor_4
    # plt.imshow(padding_img)
    # plt.show()
    p=(k-1)//2
    new_img=np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            filter_map=padding_img[padding+i-p:padding+i+p+1,padding+j-p:padding+j+p+1]
            new_img[i,j]=np.sum(filter_map*w)
    return new_img

def gaussKernel(sig,m):
    m=int(m)
    kernel=np.zeros((m,m))
    center_kernel = m // 2
    ###参考opencv
    if sig == 0:
        sig = ((m - 1) * 0.5 - 1) * 0.3 + 0.8
    sig=2*(sig**2)
    for i in range(m):
        for j in range(m):
            kernel[i,j]=np.exp(-((i-center_kernel)**2+(j-center_kernel)**2)/sig)
    return kernel/np.sum(kernel)

def session_5(img,img_path):
    a_list=[1,2,3,5]
    plt.subplot(1,len(a_list)+1,1)
    plt.imshow(img,cmap="gray")
    # print(np.rint(np.array(a_list)*3))
    kernel_size_list=np.ceil(np.array(a_list)*3)*2+1
    print(kernel_size_list)
    for index,a in enumerate(kernel_size_list):
        kernel=gaussKernel(a_list[index],a)
        filtered_img=twodConv(img, kernel,"replicate")
        plt.subplot(1,len(a_list)+1,index+2)
        plt.imshow(filtered_img,cmap="gray")
    if(not os.path.exists("./session345/")):
        os.mkdir("./session345/")
    plt.savefig("./session345/gauss_filter_mutil_sigma_"+img_path.split(".")[0]+".jpg")
    plt.show()
def session5_2(img,img_path):
    kernel = gaussKernel(1, 7)
    filtered_img = twodConv(img, kernel)
    cv2_filtered_img=cv2.GaussianBlur(img,(7,7),1)
    plt.subplot(1,3,1)
    plt.title("our mothed")
    plt.imshow(filtered_img,cmap="gray")
    plt.subplot(1,3,2)
    plt.title("opencv mothed")
    plt.imshow(cv2_filtered_img,cmap="gray")
    plt.subplot(1,3,3)
    plt.title("difference")
    plt.imshow(np.abs(cv2_filtered_img.astype(np.float32)-filtered_img.astype(np.float32)),cmap="gray")

    plt.savefig("./session345/comparsion_opencv_ourmothed_"+img_path.split(".")[0]+".jpg")
    plt.show()
def test_5_1():
    img1_path="cameraman.tif"
    img2_path="einstein.tif"
    img3_path="lena512color.tiff"
    img4_path="mandril_color.tif"

    img1=cv2.imread(img1_path,0)

    img2=cv2.imread(img2_path,0)

    img3 = cv2.cvtColor(cv2.imread(img3_path), cv2.COLOR_BGR2RGB)
    img3 = rgb1gray(img3)

    img4 = cv2.cvtColor(cv2.imread(img4_path), cv2.COLOR_BGR2RGB)
    img4 = rgb1gray(img4)

    session_5(img1,img1_path)
    session_5(img2,img2_path)
    session_5(img3,img3_path)
    session_5(img4,img4_path)


def test5_2():
    img1_path="cameraman.tif"
    img2_path="einstein.tif"
    img3_path="lena512color.tiff"
    img4_path="mandril_color.tif"

    img1=cv2.imread(img1_path,0)

    img2=cv2.imread(img2_path,0)

    img3 = cv2.cvtColor(cv2.imread(img3_path), cv2.COLOR_BGR2RGB)
    img3 = rgb1gray(img3)

    img4 = cv2.cvtColor(cv2.imread(img4_path), cv2.COLOR_BGR2RGB)
    img4 = rgb1gray(img4)
    session5_2(img1,img1_path)
    session5_2(img2,img2_path)

    session5_2(img3,img3_path)
    session5_2(img4,img4_path)
def session345_3(img_path):
    img=cv2.imread(img_path,0)
    kernel = gaussKernel(1, 7)
    filtered_img_zeropadding = twodConv(img, kernel)   #replicate
    filtered_img_replicatepadding = twodConv(img, kernel,"replicate")
    plt.subplot(1,4,1)
    plt.title("oringinal image")
    plt.imshow(img,cmap="gray")
    plt.subplot(1,4,2)
    plt.title("zero padding image")
    plt.imshow(filtered_img_zeropadding,cmap="gray")
    plt.subplot(1,4,3)
    plt.title("replicate padding image")
    plt.imshow(filtered_img_replicatepadding,cmap="gray")
    plt.subplot(1,4,4)
    plt.title("difference of two padding method  image")
    plt.imshow(np.abs(filtered_img_replicatepadding.astype(np.float32)-filtered_img_zeropadding.astype(np.float32)),cmap="gray")
    plt.savefig("./session345/"+img_path.split(".")[0]+"_compare_zero_replicate_pading.jpg")
    plt.show()

def test5_3():
    img1_path="cameraman.tif"
    img2_path="einstein.tif"
    session345_3(img1_path)
    session345_3(img2_path)

if __name__=="__main__":
    test_5_1()
    # test5_2()
    # test5_3()



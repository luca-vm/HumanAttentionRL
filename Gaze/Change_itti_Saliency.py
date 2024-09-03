import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter

##-----------------------##
"""
If you want to use this code, you can just \n
from Itti_method import Itti_Saliency_map\n
then you can just use it as :\n
itti_saliency_map = Itti_Saliency_map("your_image_path", ifshow = False)\n
the parameter ifshow controls that if the 6 + 12 + 24 different maps are displayed.\n
Hope you enjoy it!\n

Caution: Sometimes CV2 may not be able to read your image, please change the image format or check if your image path is correct.
"""
##-----------------------##

def gaussian_pyrimid(image):
   """
   return /2 resolution gaussian pyrimid
   """
   return cv2.pyrDown(image)

def eight_pyrimid_built(image):
   """
   yielding horizontal and vertical image-reduction factors ranging from 1:1 (scale zero) to 1:256 (scale eight) in eight octaves.
   """
   image_list = []
   image2 = image.copy()
   for i in range(9):
      image_list.append(image2)
      image2 = gaussian_pyrimid(image2)
   return image_list

def resize_to_normal_shape(image):
    """
    Input is provided in the form of static color images, usually digitized at 640*480 resolution
    """
    resized_image = cv2.resize(image,(640,480),interpolation=cv2.INTER_LINEAR)
    return resized_image

def seperate_RGB_chanells(image):
   """
    Seperate RGB chanells to going through following algorithms
    """
   return image[:,:,0], image[:,:,1], image[:,:,2]


def subtraction(img1,img2,ifshow = False):
    """
    return img1 - img2 but resized to the higher resolution
    """
    if ifshow:
        plt.imshow(img1)
        plt.show()
        plt.imshow(img2)
        plt.show()
    shape = img1.shape if img1.shape[0] >= img2.shape[0] else img2.shape
    shape = (shape[1],shape[0])
    image_1 = cv2.resize(img1,shape,interpolation=cv2.INTER_LINEAR)
    image_2 = cv2.resize(img2,shape,interpolation=cv2.INTER_LINEAR)
    image_1 = np.float32(image_1)
    image_2 = np.float32(image_2)
    if ifshow:
        plt.imshow(image_2)
        plt.show()
        plt.imshow(np.abs(image_1-image_2))
        plt.show()
    return image_1 - image_2 

#def processing_gabor_filters(ksize = 8,sigma = 4,lambda_ = 8,gamma = 1):
def processing_gabor_filters(ksize = 8,sigma = 4,lambda_ = 4,gamma = 1):
    """
    gengerate 4 gabor filters
    """
    kernel_0   = cv2.getGaborKernel((ksize,ksize),sigma,0,lambda_,gamma,0)
    kernel_45  = cv2.getGaborKernel((ksize,ksize),sigma,np.pi/4 ,lambda_,gamma,0)
    kernel_90  = cv2.getGaborKernel((ksize,ksize),sigma,np.pi/2 ,lambda_,gamma,0)
    kernel_135 = cv2.getGaborKernel((ksize,ksize),sigma,np.pi/4*3,lambda_,gamma,0)

    return kernel_0, kernel_45, kernel_90, kernel_135

def gabor_filter(img,k0,k45,k90,k135):
   """
   filter convlution
   """
   # tmp = [cv2.filter2D(img, cv2.CV_16SC1, k0),cv2.filter2D(img, cv2.CV_16SC1, k45),cv2.filter2D(img, cv2.CV_16SC1, k90),cv2.filter2D(img, cv2.CV_16SC1, k135)]
   return [cv2.filter2D(img, cv2.CV_16SC1, k0),cv2.filter2D(img, cv2.CV_16SC1, k45),cv2.filter2D(img, cv2.CV_16SC1, k90),cv2.filter2D(img, cv2.CV_16SC1, k135)]

def Is_scale(img_lst,c,s):
    tmp = subtraction(img_lst[c],img_lst[s])
    tmp /= np.max(np.abs(tmp))
    tmp *= 254
    # plt.imshow(tmp)
    # plt.show()
    return (np.uint8(np.where(tmp>0,tmp,0)),np.uint8(np.where(tmp<0,-tmp,0)))

def RG_scale(Rs,Gs,c,s):
    tmp = subtraction(Rs[c]-Gs[c],Gs[s]-Rs[s])
    tmp /= np.max(np.abs(tmp))
    tmp *= 254
    return (np.uint8(np.where(tmp>0,tmp,0)),np.uint8(np.where(tmp<0,-tmp,0)))

def BY_scale(Bs,Ys,c,s):
    tmp = subtraction(Bs[c]-Ys[c],Ys[s]-Bs[s])
    tmp /= np.max(np.abs(tmp))
    tmp *= 254
    return (np.uint8(np.where(tmp>0,tmp,0)),np.uint8(np.where(tmp<0,-tmp,0)))

def O_c_s_theta(Os,c,s,theta):
       tmp = np.abs(subtraction(Os[c][theta//45], Os[s][theta//45]))
       tmp /= np.max(np.abs(tmp))
       tmp *= 254
       #print(c)
       #print(s)
       #print(theta)
       #plt.imshow(tmp)
       #plt.show()
       return tmp

def normalize_img(img,M=1):
    """
    normalize img scale to 0~M, globally multiply it by (M-\\bar{m})^2
    """
    image = img / np.max(img) * M if np.max(img) else img/10
    w,h = image.shape
    maxima = maximum_filter(image, size=(w/5,h/5))
    maxima = (image == maxima)
    mnum = maxima.sum()
    maxima = np.multiply(maxima, image)
    mbar = float(maxima.sum()) / mnum if mnum else 0
    return image*((M - mbar)**2)

def addition(img1,img2,shape):
    """
    through calculate we have fourth shape
    """
    image1 = cv2.resize(img1,shape)
    image2 = cv2.resize(img2,shape)
    image1 = np.float32(image1)
    image2 = np.float32(image2)
    return subtraction(image1, -image2)

def read_image(image_path):
    image = cv2.imread(image_path)
    image = image.astype("float32")
    return image

def find_maximum(image_dict):
    """
    find maximum
    """
    maximum = -1
    for key in image_dict.keys():
        maximum = max(maximum,np.max(image_dict[key]))  
    return maximum

##--------------------------##
"""
the following function is the most important function
"""
##--------------------------##

def Itti_down_sampling(image, ifshow = False):

        
    resized_image = resize_to_normal_shape(image)

    b, g, r = seperate_RGB_chanells(resized_image)

    r_sigma = eight_pyrimid_built(r)
    g_sigma = eight_pyrimid_built(g)
    b_sigma = eight_pyrimid_built(b)
    I = [(r_sigma[i]+g_sigma[i]+b_sigma[i])/3 for i in range(9)]
    maximum = [np.max(I[i]) for i in range(9)]


    b = [np.where(b_sigma[i]>= 0.1 * maximum[i],b_sigma[i],0) for i in range(9)]
    g = [np.where(g_sigma[i]>= 0.1 * maximum[i],g_sigma[i],0) for i in range(9)]
    r = [np.where(r_sigma[i]>= 0.1 * maximum[i],r_sigma[i],0) for i in range(9)]

    Is = I
    Rs = [r[i]-(g[i]+b[i])/2 for i in range(9)]
    Gs = [g[i]-(r[i]+b[i])/2 for i in range(9)]
    Bs = [b[i]-(g[i]+r[i])/2 for i in range(9)]
    Ys = [(r[i]+g[i])/2 - np.abs(r[i] - g[i])/2 - b[i] for i in range(9)]


    kernel_0, kernel_45, kernel_90, kernel_135 = processing_gabor_filters()

    Os = [gabor_filter(Is[i],kernel_0,kernel_45,kernel_90,kernel_135) for i in range(9)]
    return Is, Rs, Gs, Bs, Ys, Os

def Itti_feature_maps(Is,Rs,Gs,Bs,Ys,Os):
    c_set = (2,3,4)
    delta_set = (3,4)
    theta_set = (0,45,90,135)

    I_dict = {}
    RG_dict = {}
    BY_dict = {}
    O_dict = {}
    for c in c_set:
        for delta in delta_set:
            I_dict[(c,c+delta)] = Is_scale(Is,c,c+delta)
            RG_dict[(c,c+delta)] = RG_scale(Rs,Gs,c,c+delta)
            BY_dict[(c,c+delta)] = BY_scale(Bs,Ys,c,c+delta)
            for theta in theta_set:
                O_dict[(c,c+delta,theta)] = O_c_s_theta(Os,c,c+delta,theta)
    return I_dict, RG_dict, BY_dict, O_dict


def Itti_motion_conspicuous_maps(fIbar,fCbar,fObar,cIbar,cCbar,cObar):
    """
    f for formal frame, and p for present frame
    we use Mp-Mf as a map to detect motion since Mp-Mf = \Delta M \times Velocity \times \Delta t 
    """
    fIbar,fCbar,fObar,cIbar,cCbar,cObar = map(lambda x: x.astype(np.int16),[fIbar,fCbar,fObar,cIbar,cCbar,cObar])
    mI_bar, mC_bar, mO_bar = map(np.abs,[fIbar-cIbar,fCbar-cCbar,fObar-cObar])
    mI_bar, mC_bar, mO_bar = map(lambda x: x.astype(np.int8),[mI_bar, mC_bar, mO_bar])
    return mI_bar, mC_bar, mO_bar

def synthesis_conspicuous_map(I_dict, RG_dict, BY_dict, O_dict,addition_shape,ifshow=False):
    c_set = (2,3,4)
    delta_set = (3,4)
    theta_set = (0,45,90,135)
    I_bar = np.zeros((1,1))
    C_bar = np.zeros((1,1))
    O_bar_0 = np.zeros((1,1))
    O_bar_45 = np.zeros((1,1))
    O_bar_90 = np.zeros((1,1))
    O_bar_135 = np.zeros((1,1))
    for c in c_set:
        for delta in delta_set:
            I_bar = addition(I_bar,normalize_img(I_dict[(c,c+delta)][0]),addition_shape)  # White
            I_bar = addition(I_bar,normalize_img(I_dict[(c,c+delta)][1]),addition_shape)  # Black
            C_bar = addition(C_bar,normalize_img(RG_dict[(c,c+delta)][0]),addition_shape) # Red
            C_bar = addition(C_bar,normalize_img(RG_dict[(c,c+delta)][1]),addition_shape) # Green
            C_bar = addition(C_bar,normalize_img(BY_dict[(c,c+delta)][0]),addition_shape) # Blue
            C_bar = addition(C_bar,normalize_img(BY_dict[(c,c+delta)][1]),addition_shape) # Yellow
            if ifshow:
                # show all adding part:
                fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3)
                ax1.imshow(I_dict[(c,c+delta)][0])
                ax1.set_title("White")
                ax4.imshow(I_dict[(c,c+delta)][1])
                ax4.set_title("Black")
                ax2.imshow(RG_dict[(c,c+delta)][0])
                ax2.set_title("Red")
                ax5.imshow(RG_dict[(c,c+delta)][1])
                ax5.set_title("Green")
                ax3.imshow(BY_dict[(c,c+delta)][0])
                ax3.set_title("Blue")
                ax6.imshow(BY_dict[(c,c+delta)][1])
                ax6.set_title("Yellow")
                plt.show()
                plt.imshow(I_bar)
                plt.show()
                plt.imshow(C_bar)
                plt.show()
            O_bar_0 = addition(O_bar_0,normalize_img(O_dict[(c,c+delta,0)]),addition_shape)
            O_bar_45 = addition(O_bar_45,normalize_img(O_dict[(c,c+delta,45)]),addition_shape)
            O_bar_90 = addition(O_bar_90,normalize_img(O_dict[(c,c+delta,90)]),addition_shape)
            O_bar_135 = addition(O_bar_135,normalize_img(O_dict[(c,c+delta,135)]),addition_shape)
    O_bar = np.zeros((1,1))
    for O_bar_theta in [O_bar_0, O_bar_45, O_bar_90, O_bar_135]:
        O_bar = addition(O_bar,normalize_img(O_bar_theta),addition_shape)
    return I_bar, C_bar, O_bar

def Itti_conspicuous_maps(image, ifshow = False):
    Is, Rs, Gs, Bs, Ys, Os = Itti_down_sampling(image,ifshow)
    I_dict, RG_dict, BY_dict, O_dict =Itti_feature_maps(Is, Rs, Gs, Bs, Ys, Os)
    I_bar, C_bar, O_bar = synthesis_conspicuous_map(I_dict, RG_dict, BY_dict, O_dict,(Is[4].shape[1],Is[4].shape[0]))

    if ifshow:
        plt.imshow(I_bar)
        plt.title("I_bar")
        plt.show()

        plt.imshow(C_bar)
        plt.title("C_bar")
        plt.show()

        plt.imshow(O_bar)
        plt.title("O_bar")
        plt.show()

    return I_bar, C_bar, O_bar

def motion_processing(image):
    resized_image = resize_to_normal_shape(image)

    b, g, r = seperate_RGB_chanells(resized_image)

    r_sigma = eight_pyrimid_built(r)
    g_sigma = eight_pyrimid_built(g)
    b_sigma = eight_pyrimid_built(b)
    I = [(r_sigma[i]+g_sigma[i]+b_sigma[i])/3 for i in range(9)]
    maximum = [np.max(I[i]) for i in range(9)]
    Is = I

    c_set = (2,3,4)
    delta_set = (3,4)
    I_dict = {}
    for c in c_set:
        for delta in delta_set:
            I_dict[(c,c+delta)] = Is_scale(Is,c,c+delta)
    I_bar = np.zeros((1,1))
    for c in c_set:
        for delta in delta_set:
            I_bar = addition(I_bar,normalize_img(I_dict[(c,c+delta)][0]),(Is[4].shape[1],Is[4].shape[0]))
            I_bar = addition(I_bar,normalize_img(I_dict[(c,c+delta)][1]),(Is[4].shape[1],Is[4].shape[0]))
    return I_bar

def Itti_Saliency_map(image):
    # image = cv2.imread("./test_jpgs/boat.jpg")
    pIs,pRs,pGs,pBs,pYs,pOs = Itti_down_sampling(image)
    I_dict, RG_dict, BY_dict, O_dict =Itti_feature_maps(pIs,pRs,pGs,pBs,pYs,pOs)
    I_bar, C_bar, O_bar = synthesis_conspicuous_map(I_dict, RG_dict, BY_dict, O_dict,(pIs[4].shape[1],pIs[4].shape[0]))
    # plt.imshow(I_bar)
    # plt.show()
    # plt.imshow(C_bar)
    # plt.show()
    # plt.imshow(O_bar)
    # plt.show()
    I_bar, C_bar, O_bar = map(normalize_img,[I_bar, C_bar, O_bar])
    image_saliency = (I_bar + C_bar + O_bar) / 3
    Saliency_map = normalize_img(image_saliency)
    Saliency_map = 255 * Saliency_map / np.max(Saliency_map)
    Saliency_map = np.uint8(Saliency_map)
    Saliency_map = cv2.resize(Saliency_map,(image.shape[1],image.shape[0]))
    # plt.imshow(Saliency_map)
    # plt.show()
    
    return Saliency_map
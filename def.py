import cv2
import glob
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd ='Tesseract-OCR/tesseract.exe'

def image_read(image):
    img=cv2.imread(image)
    return img
def de_noise(image):
    img = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    return img
def gray_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return img
def resize_image(image):
    img=image.copy()
    scale_percent = 120 # percent of original size
    width = int(img.shape[1] *(2* scale_percent) / 100)
    height = int(img.shape[0] *(1.2* scale_percent )/ 100)
    dim = (width, height)
    # resize image
    if(img.shape[1]<1500 and img.shape[0]<1000):
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_LANCZOS4)
    else:
        resized=cv2.resize(img,(img.shape[1],img.shape[0]), interpolation = cv2.INTER_LANCZOS4)
    print(resized.shape)
    return resized
def increase_contrast(image):
    #increase contrast
    bilateral_filter1 = cv2.bilateralFilter(image, 9, 75, 75)
    add_weight=cv2.addWeighted(image, 1.5, bilateral_filter1, -0.5, 0)
    bilateral_filter2=cv2.bilateralFilter(add_weight, 200, 50, 50)
    return bilateral_filter2
def reduce_light(image):
    #reduce light
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(1,1))
    cl1 = clahe.apply(image)
    return cl1
def sharpen_image(image):
    #sharpen image
    kernel_sharpening = np.array([[-1,-1,-1],
                                  [-1, 10,-1],
                                  [-1,-1,-1]])
    # applying the sharpening kernel to the input image & displaying it.
    sharpen = cv2.filter2D(image, -1, kernel_sharpening)
    #cv2.imshow('Image Sharpening', sharpen)
    return sharpen

def image_to_text(image):
    text=pytesseract.image_to_string(image,lang='eng')
    print(text)
    return text

def folder_read(path):
    images = [file for file in glob.glob(path)]
    return images

def blur_check(image):
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return  laplacian_var

def main1(i):
    img=image_read(i)
    remove_noise = de_noise(img)
    gray = gray_image(remove_noise)
    resize=resize_image(gray)
    contrast_increase=increase_contrast(resize)
    light_reducing=reduce_light(contrast_increase)
    sharpen=sharpen_image(light_reducing)
    text=image_to_text(sharpen)
    cv2.imshow("img",sharpen)
    cv2.waitKey(0)
    return text

def main2(i):
    img=image_read(i)
    resize = resize_image(img)
    gray = gray_image(resize)
    light_reducing = reduce_light(gray)
    contrast_increase = increase_contrast(light_reducing)
    contrast_increase=cv2.cvtColor(contrast_increase,cv2.COLOR_GRAY2BGR)
    remove_noise = de_noise(contrast_increase)
    sharpen = sharpen_image(remove_noise)
    text=image_to_text(sharpen)
    '''cv2.imshow("img2", sharpen)
    cv2.waitKey(0)'''
    return text

def csv_file_create(listing):
    dict2 = listing
    # field names
    with open('test2.csv', 'w') as f:
        for i in dict2:
            my_dict = i
            for key in my_dict.keys():
                f.write("%s,%s\n" % (key, my_dict[key]))
#-------------------------------------------calling of csv file function ----------------------------------------------------------


listing1=[]
list=folder_read('RC\\*.jpg')
print(list)
id=0
for i in list:
    my_dict1 = {} # stores id and similarity value from model-2
    img=image_read(i)
    blur_value=blur_check(img)
    if(blur_value<1000):
        my_dict1[id] = main1(i)
    else:
        my_dict1[id] = main2(i)
    id += 1
    listing1.append(my_dict1)

csv_file_create(listing1)

#main1()
print("======================================")
main2()
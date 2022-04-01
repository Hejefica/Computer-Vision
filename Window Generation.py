import numpy as np
import cv2

#Import Image & Translates from BGR to RGB
Img = cv2.imread("Hands.jpg")
ImgRGB = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)

#Image Color Samples
Sample1 = np.array([249, 203, 183])
Sample6 = np.array([71, 40, 32])

#Defines Upper & Lower Values
Upper = np.array(Sample1)
Lower = np.array(Sample6)

#Creates Mask Image & Contours
Mask = cv2.inRange(ImgRGB, Lower, Upper)

def Image_Labelling_Conectivity4(Image, ImageShape):
    Labeled = np.zeros(ImageShape)
    for x in range(ImageShape[0]): 
        for y in range(ImageShape[1]):
            Labeled[x,y]


    Labeled = np.uint8(Labeled)
    return Labeled

#Resize Image Output for Windowed View
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    #Initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]
    #If both the width and height are None, then return the original image
    if width is None and height is None:
        return image
    #Check to see if the width is None
    if width is None:
        #Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    #Otherwise, the height is None
    else:
        #Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    #Resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    #Return the resized image
    return resized

#Shows Images
cv2.imshow('Original Hands', image_resize(Img, height = 200))
cv2.moveWindow('Original Hands', 20, 20)
cv2.waitKey(250)

cv2.imshow('Masked Hands', image_resize(Mask, height = 200))
cv2.moveWindow('Masked Hands', 20, 285)
cv2.waitKey(0)

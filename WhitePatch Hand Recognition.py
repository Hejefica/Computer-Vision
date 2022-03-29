import numpy as np
import cv2

#Import Image
Img = cv2.imread("Hands.jpg")
Shape = Img.shape

#Image BGR Filter
def Image_ColorFilter(Image, ImageShape, Filter):
    ColorFilter = np.zeros(ImageShape)
    for x in range(Shape[0]): 
        for y in range (Shape[1]):
            if Filter == "Blue":
                ColorFilter[x,y,0] = Image[x,y,0]*1
                ColorFilter[x,y,1] = Image[x,y,1]*0.75
                ColorFilter[x,y,2] = Image[x,y,2]*0.75
            elif Filter == "Green":
                ColorFilter[x,y,0] = Image[x,y,0]*0.75
                ColorFilter[x,y,1] = Image[x,y,1]*1
                ColorFilter[x,y,2] = Image[x,y,2]*0.75
            elif Filter == "Red":
                ColorFilter[x,y,0] = Image[x,y,0]*0.75
                ColorFilter[x,y,1] = Image[x,y,1]*0.75
                ColorFilter[x,y,2] = Image[x,y,2]*1
            else:
                print("Incorrect RGB")
    ColorFilter = np.uint8(ColorFilter)
    return ColorFilter
RedImage = Image_ColorFilter(Img, Shape, "Red")
GreenImage = Image_ColorFilter(Img, Shape, "Green")
BlueImage = Image_ColorFilter(Img, Shape, "Blue")

#WhitPatch
UpperR = np.array([175, 175, 234])
UpperG = np.array([175, 234, 175])
UpperB = np.array([234, 175, 175])

def whitepatch(imagen, Upper):
    dimension = imagen.shape
    imgwp = np.empty((dimension), int)
    imagen = np.float64(imagen)
    for x in range(dimension[0]):
        for y in range (dimension[1]):
            if np.trunc(imagen[x,y,0]/Upper[0]*255) >= 255:
                imgwp[x,y,0] = 255
            else:
                imgwp[x,y,0]= np.trunc(imagen[x,y,0]/Upper[0]*255)
            if np.trunc(imagen[x,y,1]/Upper[1]*255) >= 255:
                imgwp[x,y,1] = 255
            else:
                imgwp[x,y,1]= np.trunc(imagen[x,y,1]/Upper[1]*255)

            if np.trunc(imagen[x,y,2]/Upper[2]*255) >= 255:
                imgwp[x,y,2] = 255
            else:
                imgwp[x,y,2]= np.trunc(imagen[x,y,2]/Upper[2]*255)
    imgwp= np.uint8(imgwp)
    return imgwp
WPRedImage = whitepatch(RedImage, UpperR)
WPGreenImage = whitepatch(GreenImage, UpperG)
WPBlueImage = whitepatch(BlueImage, UpperB)

#Image Color Samples
#Defines Upper Values
Sample1 = np.array([255, 249, 250])
Sample6 = np.array([71, 40, 32])
Upper = np.array(Sample1)
Lower = np.array(Sample6)

#Creates Mask Image & Contours
MaskR = cv2.inRange(WPRedImage, Lower, Upper)
MaskG = cv2.inRange(WPGreenImage, Lower, Upper)
MaskB = cv2.inRange(WPBlueImage, Lower, Upper)

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
        # calculate the ratio of the height and construct the dimensions
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

cv2.imshow('Original Hands', image_resize(Img, height = 200))
cv2.moveWindow('Original Hands', 20, 20)

cv2.imshow('Red Image', image_resize(RedImage, height = 200))
cv2.moveWindow('Red Image', 400, 20)
cv2.imshow('WhitePatch Red Image', image_resize(WPRedImage, height = 200))
cv2.moveWindow('WhitePatch Red Image', 400, 285)
cv2.imshow('WhitePatch Red Image Mask', image_resize(MaskR, height = 200))
cv2.moveWindow('WhitePatch Red Image Mask', 400,545)

cv2.imshow('Green Image', image_resize(GreenImage, height = 200))
cv2.moveWindow('Green Image', 785, 20)
cv2.imshow('WhitePatch Green Image', image_resize(WPGreenImage, height = 200))
cv2.moveWindow('WhitePatch Green Image', 785,285)
cv2.imshow('WhitePatch Green Image Mask', image_resize(MaskG, height = 200))
cv2.moveWindow('WhitePatch Green Image Mask', 785,545)

cv2.imshow('Blue Image', image_resize(BlueImage, height = 200))
cv2.moveWindow('Blue Image', 1170, 20)
cv2.imshow('WhitePatch Blue Image', image_resize(WPBlueImage, height = 200))
cv2.moveWindow('WhitePatch Blue Image', 1170, 285)
cv2.imshow('WhitePatch Blue Image Mask', image_resize(MaskB, height = 200))
cv2.moveWindow('WhitePatch Blue Image Mask', 1170,545)

cv2.waitKey(0)
import numpy as np
import cv2

#Import Image & Translates from BGR to RGB
Img = cv2.imread("Hands.jpg")
ImgRGB = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)

#Image Color Samples
Sample1 = np.array([249, 203, 183])
Sample2 = np.array([164, 88, 54])
Sample3 = np.array([190, 121, 79])
Sample4 = np.array([223, 161, 124])
Sample5 = np.array([211, 151, 108])
Sample6 = np.array([71, 40, 32])

#Defines Upper & Lower Values
Upper = np.array(Sample1)
Lower = np.array(Sample6)

#Creates Mask Image & Contours
Mask = cv2.inRange(ImgRGB, Lower, Upper)
MaskContour = cv2.inRange(ImgRGB, Lower, Upper)
Contour, Hierarchy = cv2.findContours(MaskContour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
if len(Contour) != 0:
    for Contour in Contour:
        if cv2.contourArea(Contour) > 500:
            x,y,w,h = cv2.boundingRect(Contour)
            cv2.rectangle(MaskContour, (x,y), (x+w, y+h), (255,0,0), 3)

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
cv2.waitKey(250)

cv2.imshow('Contour Hands', image_resize(MaskContour, height = 200))
cv2.moveWindow('Contour Hands', 20, 545)
cv2.waitKey(0)
import numpy as np
import cv2

#Import Image
Img = cv2.imread("Hands.jpg")
ImgRGB = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
Shape = Img.shape

#Low Images Brightnesses
def Image_Brightness(Image, ImageShape, Brightness):
    LowBright = np.zeros(ImageShape)
    for x in range(Shape[0]): 
        for y in range (Shape[1]):
            LowBright[x,y,0] = Image[x,y,0]*Brightness
            LowBright[x,y,1] = Image[x,y,1]*Brightness
            LowBright[x,y,2] = Image[x,y,2]*Brightness      
    LowBright = np.uint8(LowBright)
    return LowBright
LowBright1 = Image_Brightness(Img, Shape, 0.666)
LowBright2 = Image_Brightness(Img, Shape, 0.333)

#Chromatic Images Generation
def Image_Chromatic(ImageShape, Image):
    Chroma = np.zeros(ImageShape)
    for x in range(ImageShape[0]):
        for y in range (ImageShape[1]):
            den = int(Image[x,y,0]) + int(Image[x,y,1]) + int(Image[x,y,2])
            if den != 0:
                Chroma[x,y,0] = Image[x,y,0] / den
                Chroma[x,y,1] = Image[x,y,1] / den
                Chroma[x,y,2] = Image[x,y,2] / den
            else:
                Chroma[x,y,0] = 0
                Chroma[x,y,1] = 0
                Chroma[x,y,2] = 0
    return Chroma
Chroma1 = Image_Chromatic(Shape, Img)
Chroma2 = Image_Chromatic(Shape, LowBright1)
Chroma3 = Image_Chromatic(Shape, LowBright2)      

#Sample Mask & Mask Image Generation
UpperPercent = 0.35
LowerPercent = 0.28
Upper = np.array([int(255)*UpperPercent, int(255)*UpperPercent, int(255)*UpperPercent])
Lower = np.array([int(255)*LowerPercent, int(255)*LowerPercent, int(255)*LowerPercent])
MaskChroma1 = 255 - cv2.inRange(Chroma1*255, Lower, Upper)
MaskChroma2 = 255 - cv2.inRange(Chroma2*255, Lower, Upper)
MaskChroma3 = 255 - cv2.inRange(Chroma3*255, Lower, Upper)

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

#Shows Images
cv2.imshow('Original Hands', image_resize(Img, height = 200))
cv2.moveWindow('Original Hands', 20,20)
cv2.imshow('Original Chromatic Hands', image_resize(Chroma1, height = 200))
cv2.moveWindow('Original Chromatic Hands', 20,285)
cv2.imshow('Original Masked Hands', image_resize(MaskChroma1, height = 200))
cv2.moveWindow('Original Masked Hands', 20,545)
cv2.waitKey(250)

cv2.imshow('66.6% Brightness Hands', image_resize(LowBright1, height = 200))
cv2.moveWindow('66.6% Brightness Hands', 400,20)
cv2.imshow('66.6% Brightness Chromatic Hands', image_resize(Chroma2, height = 200))
cv2.moveWindow('66.6% Brightness Chromatic Hands', 400, 285)
cv2.imshow('66.6% Brightness Masked Hands', image_resize(MaskChroma2, height = 200))
cv2.moveWindow('66.6% Brightness Masked Hands', 400,545)
cv2.waitKey(250)

cv2.imshow('33.3% Brightness Hands', image_resize(LowBright2, height = 200))
cv2.moveWindow('33.3% Brightness Hands', 785,20)
cv2.imshow('33.3% Brightness Chromatic Hands', image_resize(Chroma3, height = 200))
cv2.moveWindow('33.3% Brightness Chromatic Hands', 785,285)
cv2.imshow('33.3% Brightness Masked Hands', image_resize(MaskChroma3, height = 200))
cv2.moveWindow('33.3% Brightness Masked Hands', 785,545)

cv2.waitKey(0)
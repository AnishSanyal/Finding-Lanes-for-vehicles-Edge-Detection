import cv2
import numpy as np
import matplotlib.pyplot as plt



def canny(image):
    gray=cv2.cvtColor(lane_image,cv2.COLOR_RGB2GRAY) #converting to gray scale
    #using gray scale for edge detection algorithm( as in to detect changes
    # in the intesity of color )

    blur = cv2.GaussianBlur(gray,(5,5),0) #Blurring to filter image noise
    #image noise can create false edges and affect edge detection

    canny = cv2.Canny(blur,50,150)    #(image,low_threshold,high_threshold)
    return canny


def display_lines(image,lines):         #Creating blue lines on the edges
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1),(x2,y2), (255,0,0),10)

    return line_image

def region_of_interest(image):      #to create a triangle or polygon
    height = image.shape[0]
    polygons = np.array([
    [(200,height),(1100,height),(550,250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(image,mask)  #computing the bitwise AND of both images ,
                                               #ultimately masking the canny image to only show the region of interest,
                                               # traced by the polygonal contour of the mask
    #return mask                   #returning a triangle on which the car will travel
    return masked_image

image=cv2.imread('test_image.jpg') #storeing the image in a numpy ndarray
lane_image = np.copy(image)  #copy of actual image
canny = canny(lane_image) #passing the image to the function
cropped_image = region_of_interest(canny)
lines = cv2.HoughLinesP(cropped_image,2, np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)     #Hough Transform is used to find the bins with most number of interception i.e the most intercepted best filter
line_image = display_lines(lane_image,lines)
combo_image = cv2.addWeighted(lane_image, 0.8,line_image,1,1) #combining the array weight of the line_image with the original image i.e adding the array of the original image to the line_image to get the result

#cv2.imshow('result',region_of_interest(cropped_image)) #window to display the image
#cv2.imshow('result',line_image)
cv2.imshow('result',combo_image)
cv2.waitKey(0) #Displays window infinitely until we push a key

#plt.imshow(canny)   #matplotlib window to plot the coordinates
#plt.show()

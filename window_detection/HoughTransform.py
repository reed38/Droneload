import cv2
import numpy as np
from math import sqrt



def getZoneArea(quadrilateral)->int:
    vertex1 = quadrilateral[0, 0]  # First vertex
    vertex2 = quadrilateral[1, 0]  # Second vertex
    vertex3 = quadrilateral[2, 0]  # Third vertex
    vertex4 = quadrilateral[3, 0]  # Fourth vertex

    side1 = sqrt((vertex1[0] - vertex4[0])**2 + (vertex1[1] - vertex4[1])**2)
    side2 = sqrt((vertex4[0] - vertex3[0])**2 + (vertex4[1] - vertex3[1])**2)

    area = side1 * side2

    return area

def keepQuadrilateral(polygonList):
    result=[]
    for polygon in polygonList:
        if(len(polygon)==4):
            result.append(polygon)
    return result


def getMainQuadrilateral(quadrilateralList, image):
    height, width, _ = image.shape
    imageArea=height*width
    areaList=[]
    for quadrilateral in quadrilateralList:
        areaList.append(getZoneArea(quadrilateral))
    
    tmpBiggestArea=0
    tmpIndexBiggestArea=0
    for areaIndex in range(len(areaList)):
        if(areaList[areaIndex]>tmpBiggestArea and areaList[areaIndex]<0.97*imageArea):
            tmpBiggestArea=areaList[areaIndex]
            tmpIndexBiggestArea=areaIndex
        
        if(areaIndex==len(areaList)-1):
            return quadrilateralList[tmpIndexBiggestArea]
    
    #if nothing is found we return by default a rectangle that encompass the image
    return np.array([[[0, 0]],
                 [[0, height]],
                 [[width, height]],
                 [[width, 0]]])

#positive number is returned if the camera is above the window, else negative if  under
def getVerticalOrientation(quadrilateral):
    vertex1 = quadrilateral[0, 0]  # First vertex
    vertex2 = quadrilateral[1, 0]  # Second vertex
    vertex3 = quadrilateral[2, 0]  # Third vertex
    vertex4 = quadrilateral[3, 0]  # Fourth vertex
    
    ##side1 = sqrt((vertex1[0] - vertex4[0])**2 + (vertex1[1] - vertex4[1])**2)
    side2 = sqrt((vertex4[0] - vertex3[0])**2 + (vertex4[1] - vertex3[1])**2)
    ##side3 = sqrt((vertex3[0] - vertex2[0])**2 + (vertex3[1] - vertex2[1])**2)
    side4 = sqrt((vertex2[0] - vertex1[0])**2 + (vertex2[1] - vertex1[1])**2)
    rapport=side2/side4
    if(rapport>=1):
        result=rapport-1
    else:
        result=-rapport
    return result

#positive number if the camera is on the left of the windows, else negative
def getHorizontalOrientation(quadrilateral):
    vertex1 = quadrilateral[0, 0]  # First vertex
    vertex2 = quadrilateral[1, 0]  # Second vertex
    vertex3 = quadrilateral[2, 0]  # Third vertex
    vertex4 = quadrilateral[3, 0]  # Fourth vertex
    
    side1 = sqrt((vertex1[0] - vertex4[0])**2 + (vertex1[1] - vertex4[1])**2)
    ##side2 = sqrt((vertex4[0] - vertex3[0])**2 + (vertex4[1] - vertex3[1])**2)
    side3 = sqrt((vertex3[0] - vertex2[0])**2 + (vertex3[1] - vertex2[1])**2)
    ##side4 = sqrt((vertex2[0] - vertex1[0])**2 + (vertex2[1] - vertex1[1])**2)
    rapport=side1/side3
    if(rapport>=1):
        result=rapport-1
    else:
        result=-rapport
    return result




def process(imagePath:str):
      
    # Load image, grayscale, Gaussian blur, Otsu's threshold
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours and approximate each polygon
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Array to store approximations
    approximations = []

    for cnt in contours:
        epsilon = 0.01*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        approximations.append(approx)


    quadrilateralList=keepQuadrilateral(approximations)

    #we will then keep the bigest image being smaller than 95% of the image size

    mainQuadrilateral=getMainQuadrilateral(quadrilateralList,image)
    
    verticalOrientation=getVerticalOrientation(mainQuadrilateral)
    horizontalHorientation=getHorizontalOrientation(mainQuadrilateral)


    # display the result on the original image
    cv2.drawContours(image, [mainQuadrilateral], 0, (255,0,0), 5)
    # Define variables for displaying the text
    text = "salutation camarade!"
    font_scale = 0.5
    thickness = 2
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font_face, font_scale, thickness)
    # Calculate the position of the top-right corner of the text box
    x = image.shape[1] - text_size[0] - 60  # 10 pixels from the right edge
    y = text_size[1] + 30  # 10 pixels from the top edge
    # Draw the text on the image
    cv2.putText(image, "vertical orientation: "+str(verticalOrientation), (x, y), font_face, font_scale, (255, 100, 0), thickness)
    cv2.putText(image, "horizontal orientation: "+str(horizontalHorientation), (x, y-20), font_face, font_scale, (255, 100, 0), thickness)


    # Display the image with detected polygons
    cv2.imshow('Polygons Detected', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


process("./image_samples/images.png")
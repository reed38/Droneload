import cv2
import numpy as np
from math import sqrt
from typing import *



def getZoneArea(quadrilateral:Tuple[Tuple[int,int],Tuple[int,int],Tuple[int,int],Tuple[int,int]])->int:
    """!Used to return the approximated area of a quadrilateral.
            ----------------
            |              |
side 1      |              |
            |              |
            ----------------
                side2
    @param quadrilateral Tuple containing 4 tuple each tuple is the coordinates of one of the quadrilateral angle.
    @return the approximated area of the quadrilateral
    """
    vertex1 = quadrilateral[0, 0]  # First vertex
    vertex2 = quadrilateral[1, 0]  # Second vertex
    vertex3 = quadrilateral[2, 0]  # Third vertex
    vertex4 = quadrilateral[3, 0]  # Fourth vertex

    side1 = sqrt((vertex1[0] - vertex4[0])**2 + (vertex1[1] - vertex4[1])**2)
    side2 = sqrt((vertex4[0] - vertex3[0])**2 + (vertex4[1] - vertex3[1])**2)
    area = side1 * side2

    return area


def keepQuadrilateral(polygonList):
    """!Go through a list of polygon, only keep and return the ones with 4 sides.
    
    @param polygonList The input list of polygons.
    @return A list of the polygon with 4 sides"""


    result=[]
    for polygon in polygonList:
        if(len(polygon)==4):
            result.append(polygon)
    return result


def getMainQuadrilateral(quadrilateralList, image:np.ndarray)->Tuple[Tuple[int,int], Tuple[int,int], Tuple[int,int], Tuple[int,int]]:
    """! Return the quadrilateral with the largest area in the list.
    
    @param quadrilateralList The list of quadrilaterals.
    @param image Image in which the quadrilateral is searched. It is used to get the size of the the zone in which the quadrilaterals have been found.
    @return The largest quadrilateral found."""

    height, width, _ = image.shape
    imageArea=height*width
    areaList=[]
    for quadrilateral in quadrilateralList:
        areaList.append(getZoneArea(quadrilateral))
    
    tmpBiggestArea=0
    tmpIndexBiggestArea=0
    for areaIndex in range(len(areaList)):
        #we check the quadrilateral is smaller than the image size. Sometime a quadrilateral is found supperposed to the image's contours
        if(areaList[areaIndex]>tmpBiggestArea and areaList[areaIndex]<0.97*imageArea):
            tmpBiggestArea=areaList[areaIndex]
            tmpIndexBiggestArea=areaIndex
        
        if(areaIndex==len(areaList)-1):
            return quadrilateralList[tmpIndexBiggestArea]
    
    #if nothing is found we return None
    return None

#positive number is returned if the camera is above the window, else negative if  under
def getVerticalOrientation(quadrilateral:Tuple[Tuple[int,int], Tuple[int,int], Tuple[int,int], Tuple[int,int]])-> int:
    """! Used to get the vertical angle of the camera relative to a window. To get the angle we use the difference in size between parallel sides
    
    @param quadrilateral: input quadrilateral.
    @return vertical angle of the camera relative to the window"""
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
def getHorizontalOrientation(quadrilateral:Tuple[Tuple[int,int], Tuple[int,int], Tuple[int,int], Tuple[int,int]]):
    """! Used to get the horizontal angle of the camera relative to a window. To get the angle we use the difference in size between parallel sides.
        @param quadrilateral: input quadrilateral.
        @return vertical angle of the camera relative to the window.
    """
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




def detectRectangle(imagePath:str,debug:bool)->Tuple[np.ndarray, int, int] :
      
    # Load image, grayscale, Gaussian blur, Otsu's threshold
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find the contours present in the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Array to store approximations
    approximations = []

    #Approximating the contours to polygons
    for cnt in contours:
        epsilon = 0.01*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        approximations.append(approx)

    #keeping only the quadrilaterals among the polygons
    quadrilateralList=keepQuadrilateral(approximations)

    #we will then keep the largest quadrilateral (being smaller than the size of the image)
    mainQuadrilateral=getMainQuadrilateral(quadrilateralList,image)
    
    #Getting vertical and horieontal angle of the camera relative to the window
    verticalOrientation=getVerticalOrientation(mainQuadrilateral)
    horizontalHorientation=getHorizontalOrientation(mainQuadrilateral)


    # display the result on the original image if debug is activated
    if(debug):
        cv2.drawContours(image, [mainQuadrilateral], 0, (255,0,0), 5)
        # Define variables for displaying the text
        font_scale = 0.5
        thickness = 2
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        # Calculate the position of the top-right corner of the text box
        x = image.shape[0]-400  # 10 pixels from the right edge
        y = image.shape[1]-200   # 10 pixels from the top edge
        # Draw the text on the image
        cv2.putText(image, "vertical orientation: "+str(verticalOrientation), (x, y), font_face, font_scale, (255, 100, 0), thickness)
        cv2.putText(image, "horizontal orientation: "+str(horizontalHorientation), (x, y-20), font_face, font_scale, (255, 100, 0), thickness)


    return image, verticalOrientation, horizontalHorientation




def test():
    image,_,_=detectRectangle("./image_samples/images.png",debug=True)
    cv2.imshow('Polygons Detected', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


test()
import cv2
import numpy as np



def find_intersection(line1, line2):
    # Unpack the endpoints of the lines
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    # Calculate the slopes and intercepts of the lines
    m1 = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else float('inf')
    m2 = (y4 - y3) / (x4 - x3) if x4 - x3 != 0 else float('inf')
    
    b1 = y1 - m1 * x1 if m1 != float('inf') else x1
    b2 = y3 - m2 * x3 if m2 != float('inf') else x3
    
    # Handle the case where the lines are parallel
    if m1 == m2:
        return None
    
    # Calculate the intersection point
    if m1 == float('inf'):
        x_intersect = x1
        y_intersect = m2 * x1 + b2
    elif m2 == float('inf'):
        x_intersect = x3
        y_intersect = m1 * x3 + b1
    else:
        x_intersect = (b2 - b1) / (m1 - m2)
        y_intersect = m1 * x_intersect + b1
    
    return x_intersect, y_intersect


def getHorizontalDistance(line,image):
    height, width, _ = image.shape
    horizontal_line=[0,int(height/2),int(width),int(height/2)]
    intersection=find_intersection(line,horizontal_line)
    return width/2-intersection[0]

def getAngle(line):
 
    x1, y1, x2, y2 = line
    tanAngle = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else float('inf')
    # Calculate the angle each line makes with the x-axis
    angle = np.arctan(tanAngle)
    return np.degrees(angle)
    


def getLine(image_path:str,debug:bool):
    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=10, maxLineGap=10)

    # Find the thickest line
    thickest_line = None
    thickest_length = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate the length of the line
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length > thickest_length:
            thickest_length = length
            thickest_line = line

    # Draw only the thickest line
    if thickest_line is not None:
        x1, y1, x2, y2 = thickest_line[0]
        # Extend the line beyond its endpoints
        extended_line = [
            int(x1 - 10 * (x2 - x1)),  # Extend 10 times the length of the line in the x1 direction
            int(y1 - 10 * (y2 - y1)),  # Extend 10 times the length of the line in the y1 direction
            int(x2 + 10 * (x2 - x1)),  # Extend 10 times the length of the line in the x2 direction
            int(y2 + 10 * (y2 - y1))   # Extend 10 times the length of the line in the y2 direction
        ]
    
    
        cv2.line(image, (extended_line[0], extended_line[1]), (extended_line[2], extended_line[3]), (0, 255, 0), 2)
        height, width, _ = image.shape
        horizontal_line=[0,int(height/2),int(width),int(height/2)]
        centerPoint=(int(width/2),int(height/2))
        cv2.line(image, (horizontal_line[0], horizontal_line[1]), (horizontal_line[2], horizontal_line[3]), (0, 0, 255), 2)
        cv2.circle(image, centerPoint, 10, (0, 0, 255 ), -1)
        # Display result
    if(debug):
        cv2.imshow('Prolongated Line', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return extended_line,image




def testFindAngle(filePath:str):
    line,image=getLine(filePath,debug=False)
    angle=getAngle(line=line)
    horizontalOffset=getHorizontalDistance(line=line,image=image)
    # Define variables for displaying the text
    font_scale = 0.5
    thickness = 2
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    # Calculate the position of the top-right corner of the text box
    x = 300  # 10 pixels from the right edge
    y = 300 # 10 pixels from the top edge
    # Draw the text on the image
    cv2.putText(image, "horizontal Offset: "+str(horizontalOffset), (x, y), font_face, font_scale, (255, 100, 0), thickness)
    cv2.putText(image, "angle offset: "+str(angle), (x, y-20), font_face, font_scale, (255, 100, 0), thickness)
    cv2.imshow('line detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



image_path = "./images_samples/image_2.png"
testFindAngle(filePath=image_path)
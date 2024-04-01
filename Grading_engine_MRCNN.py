import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from skimage import morphology
import math
import statistics

#%%

def get_boundary(image):
    
    shape = image.shape
    X_boundary = []
    Y_boundary = []
    Boundary_Coordinates =[]
    for i in range(shape[0]):
        for j in range(shape[1]):
            neighbours = [[i-1,j-1],[i-1,j],[i-1,j+1],[i,j-1],[i,j+1],[i+1,j-1],[i+1,j],[i+1,j+1]]
            x = image[i,j]
            if x>0:
                for neighbour in neighbours:
                    if image[neighbour[0], neighbour[1]] == 0:
                        X_boundary.append(i)  ## row
                        Y_boundary.append(j)  ## column
                        Boundary_Coordinates.append([i,j])
                        break
    
    return Boundary_Coordinates
#%%

def major_axis_points(points: list, coefficient: float, intercept: float,
                 midpoint: float, position: str):
    
    if position == 'Head':
        filter_condition = lambda p: p[0] <= midpoint
    elif position == 'Apex':
        filter_condition = lambda p: p[0] > midpoint
    else:
        raise ValueError("Invalid position value")

    relevant_points = filter(filter_condition, points)

    minum = float('inf')
    coordinate = None

    for point in relevant_points:
        x, y = point
        Y = coefficient * x + intercept
        difference = abs(y - Y)

        if  difference < minum:
            minum = difference
            coordinate = point


    return coordinate, minum

#%%

def minor_points(points: list, x:float, y:float, coefficient: float,
                 minor_midpoint: float, position: str):
    
    if position == 'bottom':
        filter_condition = lambda p: p[1] <= minor_midpoint
    elif position == 'top':
        filter_condition = lambda p: p[1] > minor_midpoint
    else:
        raise ValueError("Invalid position value")

    relevant_points = filter(filter_condition, points)

    minum = float('inf')
    coordinate = None
    intcpt = 0

    for point in relevant_points:
        X, Y = point
        c = y - (-1/coefficient)*x
        Y1 = (-1/coefficient)*X + c #eqn of line perpendicular to axis and passing through (x,y)
        
        #Y = (-1/coefficient) * (X-x) + y   #eqn of line perpendicular to axis and passing through (x,y)

        #print('Equation of perpendicular line: y = {0:.4f}* x + {1:.2f}'.format((-1/coefficient), y))

        difference = abs(Y - Y1)

        if  difference < minum:
            minum = difference
            coordinate = point
            intcpt = c
    
    return coordinate, intcpt

#%%

def minor_axis_points(Boundary:list, Head:list, Apex: list, coefficient: float,intercept:float, minor_midpoint):
    
    maxm = float('-inf')
    minor_lengths = []
    for i in range(Head[0], Apex[0]):
        j = coefficient * i + intercept
       
        top, intcpt = minor_points(Boundary, i, j, coefficient, minor_midpoint, 'top')
        bottom, intcpt = minor_points(Boundary, i, j, coefficient, minor_midpoint, 'bottom')
        minor_length =  math.sqrt((top[0]- bottom[0])**2 + (top[1]-bottom[1])**2)          #top[1] - bottom[1]
        #print(minor_length)
        minor_lengths.append([top[0],minor_length])
        if minor_length > maxm:
            maxm = minor_length
            minor_points_coordinate = [top, bottom]
            intcpt1 = intcpt
    return minor_points_coordinate, maxm, intcpt1

#%%

def resize(image, scale):
    scale_percent = scale # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
  
    # resize image
    img = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return img

#%%

def  shape_identification(Boundary_Coordinates, major_axis, minor_axis):
    df = pd.DataFrame(Boundary_Coordinates)
    x = df.iloc[:,0].to_numpy().reshape(-1,1) #np.array(X_boundary).reshape(-1,1)  ## rows value
    y = df.iloc[:,1].to_numpy().reshape(-1,1)
    upper_points = []
    lower_points = []
    for i in Boundary_Coordinates:
        len_minor_axis_to_apex = major_axis[1][0] - minor_axis[0][0]
        
        if i[0] > minor_axis[0][0] and i[0]< (major_axis[1][0]-len_minor_axis_to_apex* 0.2) :
                   if i[1]> major_axis[1][1]:
                       upper_points.append(i)

        if i[0] > minor_axis[1][0] and i[0]< (major_axis[1][0]-len_minor_axis_to_apex* 0.2):
            if i[1]< major_axis[1][1]:
                lower_points.append(i)
                
    df1 = pd.DataFrame(upper_points)
    X1 = df1[0].to_numpy().reshape(-1,1)
    Y1 = df1[1].to_numpy().reshape(-1,1)
    
    df2 = pd.DataFrame(lower_points)
    X2 = df2[0].to_numpy().reshape(-1,1)
    Y2 = df2[1].to_numpy().reshape(-1,1)
    
    
    regr = LinearRegression(fit_intercept = True)
    
    regr.fit(X1,Y1)
    coefficient1 = regr.coef_[0][0]
    intercept1 = regr.intercept_[0]
    r1 =  regr.score(X1,Y1)

    y1 = coefficient1* upper_points[0][0] + intercept1
    y11 = coefficient1* upper_points[-1][0] + intercept1
    regr.fit(X2,Y2)
    coefficient2 = regr.coef_[0][0]
    intercept2 = regr.intercept_[0]
    r2 =  regr.score(X2,Y2)

    y2 = coefficient2* lower_points[1][0] + intercept2
    y22 = coefficient2* lower_points[-1][0] + intercept2
    
    '''
    plt.scatter(y,x, s = 5, color="black")
    plt.plot([y1, y11], [upper_points[0][0], upper_points[-1][0]], c = 'red',linewidth = 2 )
    plt.plot([y2, y22], [lower_points[0][0], lower_points[-1][0]], c = 'red',linewidth = 2 )
                                                        
    plt.scatter(Y1, X1, s = 1, color="blue")
    plt.scatter(Y2, X2, s =1, color="blue")
    plt.show()
    '''
    Upper_angle = math.degrees(math.atan(abs(coefficient1)))
    Lower_angle = math.degrees(math.atan(abs(coefficient2)))
    Total_angle = round((Upper_angle + Lower_angle),2)
    #angles.append([ImageName, Upper_angle, Lower_angle, Total_angle, r1, r2, major_axis_length, minor_axis_length])
    #print('Regression equation: y = {0:.4f}* x + {1:.2f}'.format(coefficient1, intercept1))
    print('Apex angle: {:.2f}'.format(Total_angle))
    
    return Upper_angle, Lower_angle, Total_angle, r1, r2

#%%


def zero_padding(image, buffer:int):
    
    if buffer%2 != 0:
        return print('Give even values for buffer')
    else:
        padded_image = np.zeros((image.shape[0]+ 2*buffer, image.shape[1]+ 2*buffer))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                padded_image[i+(buffer+1)][j+(buffer+1)] = image[i][j]
        return padded_image
    
    

#%%

def shape_classification(ratio, angle, r2):
    
    """
    Classify a shape based on given parameters.

    Parameters:
    - ratio (float): The ratio of major axis to minor axis length parameter.
    - angle (float): The angle of strawberry sides parameter.
    - r2 (float): The coeffictient of determination (R2) of the points on sides of strawberry.

    Returns:
    - str: The classified shape.
    """
    
    if ratio >= 1.5 and angle>20:
         shape = 'Long taper'
         print('Ratio: {}'.format(ratio))
         print(shape)
    elif angle <=35 and r2<=1.0:
        shape = 'Square'
        print('R2: {}'.format(r2))
        print(shape)
    elif angle>35 and r2<0.9:
        shape = 'Round'
        print('R2: {}'.format(r2))
        print(shape)
    elif angle>=35 and r2>=0.9:
        shape = 'Taper'
        print('R2: {}'.format(r2))
        print(shape)

    return shape

#%%

def ripeness(Boundary:list, Head:list, Apex: list, coefficient: float,intercept:float, minor_midpoint, image):
    
    rotated_img =  cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    flipped_img = cv2.flip(rotated_img, 0)

    Ripe_length = 0
    Ripe_percentage = 0
    major_axis_length = math.sqrt((Head[0]- Apex[0])**2 + (Head[1]-Apex[1])**2)
    height, width = np.shape(flipped_img)
    for i in range(Head[0], Apex[0]):

        j = coefficient * i + intercept
       
        top, intcpt = minor_points(Boundary, i, j, coefficient, minor_midpoint, 'top')
        bottom, intcpt = minor_points(Boundary, i, j, coefficient, minor_midpoint, 'bottom')
        minor_length =  math.sqrt((top[0]- bottom[0])**2 + (top[1]-bottom[1])**2)          #top[1] - bottom[1]
        #print(minor_length)
        Values = []
        ripeness_boundary = []
        Ripeness = None
        for Y in range(bottom[1], (top[1])):
            #print(Y)
            X = round((intcpt-Y)*coefficient)
            #print(Y,X)
            value = flipped_img[Y,X]
            #print(value)
            if value>=130:
                Values.append(value)
        
        ratio = len(Values)/minor_length

        if ratio>=0.4:
            Ripe_length = math.sqrt((i- Apex[0])**2 + (j-Apex[1])**2)
            Ripe_percentage = (Ripe_length/major_axis_length)*100
            
            if Ripe_percentage >= 70:
                Ripeness = 'Ripe'
            elif 70>Ripe_percentage >= 40:
                Ripeness = 'Semi-ripe'
            else: 
                Ripeness = 'Unripe'
            
            bottom = [bottom[0]-10, bottom[1]-10] ##Removing the padding
            top = [top[0]-10, top[1]-10]
            ripeness_boundary = [bottom, top]
            break
        else:
            Ripeness = 'Unripe'
            Ripe_percentage = 0
            ripeness_boundary = None
        
    
    return  [Ripeness, Ripe_percentage], ripeness_boundary
     
#%%

def axes(img): 
    
    img = zero_padding(img, 10)
    

    Boundary_Coordinates = get_boundary(img)

    df = pd.DataFrame(Boundary_Coordinates)

    x = df.iloc[:,0].to_numpy().reshape(-1,1) #np.array(X_boundary).reshape(-1,1)  ## rows value
    y = df.iloc[:,1].to_numpy().reshape(-1,1) #np.array(Y_boundary).reshape(-1,1)  ## column value
    midpoint = min(df.iloc[:,0])+ (max(df.iloc[:,0]) - min(df.iloc[:,0]))/2
    minor_midpoint = min(df.iloc[:,1]) + (max(df.iloc[:,1]) - min(df.iloc[:,1]))/2

    #%%            
    regr = LinearRegression(fit_intercept = True)
    regr.fit(x,y)
    coefficient = regr.coef_[0][0]
    intercept = regr.intercept_[0]
    
    #print('Regression equation: y = {0:.4f}* x + {1:.2f}'.format(regr.coef_[0][0], regr.intercept_[0]))

    #%%
    Head, minum1 = major_axis_points(Boundary_Coordinates, coefficient, intercept, midpoint, 'Head')  
    Apex, minum2 = major_axis_points(Boundary_Coordinates, coefficient, intercept, midpoint, 'Apex') 
    major_axis = [Head, Apex, coefficient, intercept]
    
    Head = [Head[0]-10, Head[1]-10]  ##Removing the pading
    Apex = [Apex[0]-10, Apex[1]-10]  ##Removing the padding
      
    major_axis_length = math.sqrt((Head[0]- Apex[0])**2 + (Head[1]-Apex[1])**2)  #Apex[0] - Head[0]
    minor_axis, minor_axis_length, intcpt1 = minor_axis_points(Boundary_Coordinates, Head, Apex, coefficient, intercept, minor_midpoint)
    #print('Regression equation: y = {0:.4f}* x + {1:.2f}'.format((-1/coefficient), intcpt1))
    
    #fig = plt.figure(dpi = 500)
    #plt.scatter(x,y, s = 5, color="black")
    #plt.imshow(img, cmap = 'gray')
    #plt.plot([Head[1], Apex[1]], [Head[0], Apex[0]], c = 'green',linewidth = 2 )
    #plt.plot([minor_axis[1][1],minor_axis[0][1]], [minor_axis[1][0],minor_axis[0][0]],
             #c = 'purple',linewidth = 2 )    
    #plt.axis('off')
    #plt.show()
    #plt.show()
    return Boundary_Coordinates, major_axis, minor_axis, major_axis_length, minor_axis_length

#%%

def get_attributes(image, mask):
    

    Boundary_Coordinates, major_axis, minor_axis, major_axis_length, minor_axis_length = axes(mask)
    
    Upper_angle, Lower_angle, Total_angle, r1, r2 = shape_identification(Boundary_Coordinates, major_axis, minor_axis)
    major_to_minor_ratio = round((major_axis_length/ minor_axis_length),2)
    average_R2 = round(statistics.mean([r1, r2]), 2)    
    Shape = shape_classification(major_to_minor_ratio, Total_angle, average_R2)
    
    #mask = zero_padding(mask_binary, 10).astype('uint8')
    #RGB_image = image[x1:x2, y1:y2,:]
    #RGB_img = cv2.bitwise_and(image, image, mask = mask_binary)
    
    Lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)[:,:,1]
    Lab_padded = zero_padding(Lab, 10)
    Head = major_axis[0]
    Apex = major_axis[1]
    coefficient = major_axis[2]
    intercept = major_axis[3]
    
    df = pd.DataFrame(Boundary_Coordinates)
    minor_midpoint = min(df.iloc[:,1]) + (max(df.iloc[:,1]) - min(df.iloc[:,1]))/2
    
    Ripeness, ripeness_boundary = ripeness(Boundary_Coordinates, Head, Apex, coefficient, intercept, minor_midpoint,Lab_padded )
    
    Head = [Head[0]-10, Head[1]-10]
    Apex = [Apex[0]-10, Apex[1]-10]
    
    updated_major_axis = [Head, Apex]
    
    
    
    
    
    #working_distance = Depth*1000
    #major_length = 0.0010217* major_axis_length*working_distance
    
    return [major_axis_length, updated_major_axis], Shape, [Ripeness, ripeness_boundary]

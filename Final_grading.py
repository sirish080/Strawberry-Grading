import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def shape_score(shape:str):
    
    if shape == 'Taper':
        return 1
    elif shape == 'Long taper':
        return 2
    elif shape == 'Round':
        return 3
    else:
        return 4

def ripeness_score(ripeness:str):
    if ripeness == 'Ripe':
        return 1
    elif ripeness == 'Semi':
        return 2
    else:
        return 3

def size_score(size):
    if 45 <= size < 50:
        return 1
    elif 35 <= size < 45:
        return 2
    elif size >= 50:
        return 3
    else:
        return 4
    
    

def grading(attributes):
    
    Shape = attributes[0]
    Ripeness = attributes[1]
    Size = attributes[2]
    
    Shape_weight = 0.33
    Ripeness_weight = 0.33
    Size_weight = 0.33
    Final_score = Shape_weight * shape_score(Shape) + Size_weight * size_score(Size) + Ripeness_weight * ripeness_score(Ripeness)
    
    if Final_score < 2:
        return 1, Final_score, [shape_score(Shape), size_score(Size), ripeness_score(Ripeness)]
    elif 2 <= Final_score < 3:
        return 2,Final_score, [shape_score(Shape), size_score(Size), ripeness_score(Ripeness)]
    else:
        return 3,Final_score, [shape_score(Shape), size_score(Size), ripeness_score(Ripeness)]


               
    
    
    
    
    
    
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 11:18:26 2017

@author: alistairgj
"""

import math

x1 = float(input('Enter x1'))
y1 = float(input('Enter y1'))

x2 = float(input('Enter x2'))
y2 = float(input('Enter y2'))

x3 = float(input('Enter x3'))
y3 = float(input('Enter y3'))

#def delta_xi(x1, x2):
#    return abs(x2-x1)

#def delta_xii(x2, x3):
#    return abs(x3-x2)

#def delta_yi(y1, y2):
#    return abs(y2-y1)

#def delta_yii(y2, y3):
#    return abs(y3-y2)

#def delta(value1, value2):
#    return abs(value2 - value1)

def calc_side(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2+(y2-y1)**2)

side1 = calc_side(x1, y1, x2, y2)
side2 = calc_side(x2, y2, x3, y3)
side3 = calc_side(x1, y1, x3, y3)
    
def perimeter(side1, side2, side3):
    return side1 + side2 + side3

print perimeter(side1, side2, side3)



def calc_perimeter(delta_xi, delta_xii, delta_yi, delta_yii):
    return math.sqrt(delta_xi**2 + delta_xii**2 + delta_yi**2 + delta_yii**2) + delta_xi + delta_xii + delta_yi + delta_yii


print()






# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains geometry functions that relate with Part1 in MP2.
"""

import math
import numpy as np
from const import *

def computeCoordinate(start, length, angle):
    """Compute the end cooridinate based on the given start position, length and angle.

        Args:
            start (tuple): base of the arm link. (x-coordinate, y-coordinate)
            length (int): length of the arm link
            angle (int): degree of the arm link from x-axis to couter-clockwise

        Return:
            End position (int,int):of the arm link, (x-coordinate, y-coordinate)
    """
    radians = math.radians(angle)
    x = int(start[0] + length * math.cos(radians))
    y = int(start[1] - length * math.sin(radians))
    return (x,y)

# p1 and p2 are the start and end of line and p3 is obstacle
# I ADAPTED THIS FORMULA FROM THE SOURCE BELOW
# https://stackoverflow.com/questions/6068660/checking-a-line-segment-is-within-a-distance-from-a-point
def distance_from_point_to_segment(start, end, obstacle):
    v = (end[0] - start[0], end[1] - start[1])   # vector between start and obstacle
    w = (obstacle[0] - start[0], obstacle[1] - start[1])  # vector between end and obstacle
    c1 = float((v[0] * w[0]) + (v[1] * w[1]))  # dot product between v and w vectors
    c2 = float((v[0] * v[0]) + (v[1] * v[1]))  # dot product of v with itself
    b = float(c1 / c2)
    newPoint = (start[0] + (b * v[0]), start[1] + (b * v[1]))

    if c1 <= 0:
        return math.hypot(obstacle[0] - start[0], obstacle[1] - start[1])

    if c2 <= c1:
        return math.hypot(obstacle[0] - end[0], obstacle[1] - end[1])

    return math.hypot(obstacle[0] - newPoint[0], obstacle[1] - newPoint[1])

def doesArmTouchObjects(armPosDist, objects, isGoal=False):
    """Determine whether the given arm links touch any obstacle or goal

        Args:
            armPosDist (list): start and end position and padding distance of all arm links [(start, end, distance)]
            objects (list): x-, y- coordinate and radius of object (obstacles or goals) [(x, y, r)]
            isGoal (bool): True if the object is a goal and False if the object is an obstacle.
                           When the object is an obstacle, consider padding distance.
                           When the object is a goal, no need to consider padding distance.
        Return:
            True if touched. False if not.
    """
    for object in objects:
        obstacle = (object[0], object[1])
        radius = object[2]
        for arm in armPosDist:
            arm_start = arm[0]
            arm_end = arm[1]
            arm_padding = arm[2]
            d = distance_from_point_to_segment(arm_start, arm_end, obstacle)
            if (isGoal and d <= radius and not doesArmTipTouchGoals(arm_end, [object])) or (not isGoal and d <= (radius + arm_padding)):
                return True
    return False

def doesArmTipTouchGoals(armEnd, goals):
    """Determine whether the given arm tip touch goals

        Args:
            armEnd (tuple): the arm tip position, (x-coordinate, y-coordinate)
            goals (list): x-, y- coordinate and radius of goals [(x, y, r)]. There can be more than one goal.
        Return:
            True if arm tick touches any goal. False if not.
    """
    for goal in goals:
        if math.hypot(armEnd[0] - goal[0], armEnd[1] - goal[1]) <= goal[2]:
            return True
    return False


def isArmWithinWindow(armPos, window):
    """Determine whether the given arm stays in the window

        Args:
            armPos (list): start and end positions of all arm links [(start, end)]
            window (tuple): (width, height) of the window

        Return:
            True if all parts are in the window. False if not.
    """
    width, height = window[0], window[1]
    for arm in armPos:
        x1, x2, y1, y2 = arm[0][0], arm[1][0], arm[0][1], arm[1][1]
        if x1 < 0 or x1 > width or x2 < 0 or x2 > width or y1 < 0 or y1 > height or y2 < 0 or y2 > height:
            return False
    return True

if __name__ == '__main__':
    computeCoordinateParameters = [((150, 190),100,20), ((150, 190),100,40), ((150, 190),100,60), ((150, 190),100,160)]
    resultComputeCoordinate = [(243, 156), (226, 126), (200, 104), (57, 156)]
    testRestuls = [computeCoordinate(start, length, angle) for start, length, angle in computeCoordinateParameters]
    assert testRestuls == resultComputeCoordinate

    testArmPosDists = [((100,100), (135, 110), 4), ((135, 110), (150, 150), 5)]
    testObstacles = [[(120, 100, 5)], [(110, 110, 20)], [(160, 160, 5)], [(130, 105, 10)]]
    resultDoesArmTouchObjects = [
        True, True, False, True, False, True, False, True,
        False, True, False, True, False, False, False, True
    ]

    testResults = []
    for testArmPosDist in testArmPosDists:
        for testObstacle in testObstacles:
            testResults.append(doesArmTouchObjects([testArmPosDist], testObstacle))
            # print(testArmPosDist)
            # print(doesArmTouchObjects([testArmPosDist], testObstacle))

    print("\n")
    for testArmPosDist in testArmPosDists:
        for testObstacle in testObstacles:
            testResults.append(doesArmTouchObjects([testArmPosDist], testObstacle, isGoal=True))
            # print(testArmPosDist)
            # print(doesArmTouchObjects([testArmPosDist], testObstacle, isGoal=True))

    assert resultDoesArmTouchObjects == testResults

    testArmEnds = [(100, 100), (95, 95), (90, 90)]
    testGoal = [(100, 100, 10)]
    resultDoesArmTouchGoals = [True, True, False]

    testResults = [doesArmTipTouchGoals(testArmEnd, testGoal) for testArmEnd in testArmEnds]
    assert resultDoesArmTouchGoals == testResults

    testArmPoss = [((100,100), (135, 110)), ((135, 110), (150, 150))]
    testWindows = [(160, 130), (130, 170), (200, 200)]
    resultIsArmWithinWindow = [True, False, True, False, False, True]
    testResults = []
    for testArmPos in testArmPoss:
        for testWindow in testWindows:
            testResults.append(isArmWithinWindow([testArmPos], testWindow))
    assert resultIsArmWithinWindow == testResults

    print("Test passed\n")

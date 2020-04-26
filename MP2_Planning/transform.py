
# transform.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the transform function that converts the robot arm map
to the maze.
"""
import copy
from arm import Arm
from maze import Maze
from search import *
from geometry import *
from const import *
from util import *

def transformToMaze(arm, goals, obstacles, window, granularity):
    """This function transforms the given 2D map to the maze in MP1.

        Args:
            arm (Arm): arm instance
            goals (list): [(x, y, r)] of goals
            obstacles (list): [(x, y, r)] of obstacles
            window (tuple): (width, height) of the window
            granularity (int): unit of increasing/decreasing degree for angles

        Return:
            Maze: the maze instance generated based on input arguments.

    """
    alpha_range = arm.getArmLimit()[0]  # max and min values for alpha and beta angles
    alpha_start = arm.getArmAngle()[0]  # initial alpha and beta angle positions
    beta_range = (0, 0)
    beta_start = 0
    gamma_range = (0, 0)
    gamma_start = 0

    rows = 1 + int(((alpha_range[1] - alpha_range[0]) / granularity))
    columns = 1
    third = 1

    if len(arm.getArmPosDist()) >= 2:
        beta_range = arm.getArmLimit()[1]
        beta_start = arm.getArmAngle()[1]
        columns = 1 + int(((beta_range[1] - beta_range[0]) / granularity))
    if len(arm.getArmPosDist()) >= 3:
        gamma_range = arm.getArmLimit()[2]
        gamma_start = arm.getArmAngle()[2]
        third = 1 + int(((gamma_range[1] - gamma_range[0]) / granularity))

    maze = [[[0 for j in range(third)] for i in range(columns)] for y in range(rows)] # initializing maze array with 0

    for i in range(alpha_range[0], alpha_range[1] + 1, granularity):
        for j in range(beta_range[0], beta_range[1] + 1, granularity):
            for k in range(gamma_range[0], gamma_range[1] + 1, granularity):
                arm.setArmAngle((i, j, k))
                armPosDist = arm.getArmPosDist() # (start, end, distance) for all arm links
                armPos = arm.getArmPos() # (start, end) for all arm links
                armTip = arm.getEnd() # (x, y) for arm tip
                index = angleToIdx([i, j, k], [alpha_range[0], beta_range[0], gamma_range[0]], granularity)
                x, y, z = index[0], index[1], index[2]
                if i == alpha_start and j == beta_start and k == gamma_start:
                    maze[x][y][z] = 'P'
                elif doesArmTouchObjects(armPosDist, obstacles, False) or doesArmTouchObjects(armPosDist, goals, True) or not isArmWithinWindow(armPos, window):
                    maze[x][y][z] = '%'
                elif doesArmTipTouchGoals(armTip, goals):
                    maze[x][y][z] = '.'
                elif isArmWithinWindow(armPos, window):
                    maze[x][y][z] = ' '

    return Maze(maze, [alpha_range[0], beta_range[0], gamma_range[0]], granularity)

# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
from heapq import heappop, heappush

class bfsNode:
    def __init__(self, pos):
        self.pos = pos
        self.parent = []

def search(maze, searchMethod):
    return {
        "bfs": bfs,
    }.get(searchMethod, [])(maze)

def bfs(maze):
    # Write your code here
    """
    This function returns optimal path in a list, which contains start and objective.
    If no path found, return None.
    """
    start = maze.getStart()
    end = set(maze.getObjectives())
    q = deque()
    start_node = bfsNode(start)
    start_node.parent = None
    visited = set()
    q.append(start_node)
    visited.add(start)
    ret = []

    while q:
        curr = q.popleft()

        if curr.pos in end:
            while curr:
                ret.append(curr.pos)
                curr = curr.parent
            return ret[::-1]

        curr_neighbors = maze.getNeighbors(curr.pos[0], curr.pos[1], curr.pos[2])

        for neighbor in curr_neighbors:
            if neighbor not in visited:
                neighbor_node = bfsNode(neighbor)
                neighbor_node.parent = curr
                visited.add(neighbor_node.pos)
                q.append(neighbor_node)

    return None

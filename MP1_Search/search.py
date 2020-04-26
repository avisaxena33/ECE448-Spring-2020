# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,extra)

#Import statements
import collections
import heapq
import copy
import timeit

#Small node class to keep track of current cell position and the shortest path from start to current cell
class bfsNode:
    def __init__(self, pos):
        self.pos = pos
        self.parent = []

#aStar A -> B class
class starNode:
    def __init__(self, pos):
        self.pos = pos
        self.g = 0
        self.h = 0
        self.f = 0
        self.parent = None

#aStar corners class
class starNodeMultiple:
    def __init__(self, pos):
        self.pos = pos
        self.g = 0
        self.h = 0
        self.f = 0
        self.targets = set()
        self.currentTarget = None
        self.seen = set()
        self.parent = None

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "extra": extra,
    }.get(searchMethod)(maze)


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    end = maze.getObjectives()[0]
    q = collections.deque()
    start_node = bfsNode(start)
    start_node.parent = None
    visited = set()
    q.append(start_node)
    visited.add(start)
    ret = []

    while q:
        curr = q.popleft()

        if curr.pos == end:
            while curr:
                ret.append(curr.pos)
                curr = curr.parent
            return ret[::-1]

        curr_neighbors = maze.getNeighbors(curr.pos[0], curr.pos[1])

        for neighbor in curr_neighbors:
            if neighbor not in visited:
                neighbor_node = bfsNode(neighbor)
                neighbor_node.parent = curr
                visited.add(neighbor_node.pos)
                q.append(neighbor_node)

    return []


def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start, end = maze.getStart(), maze.getObjectives()[0]
    start_node, end_node = starNode(start), starNode(end)
    start_node.g = start_node.h = start_node.f = end_node.g = end_node.h = end_node.f = 0
    open_list, closed_list = [], set()
    heapq.heappush(open_list, (start_node.f, start_node.pos, start_node))
    ret = []
    closed_list.add(start_node.pos)

    while open_list:
        curr = heapq.heappop(open_list)[2]

        if curr.pos == end_node.pos:
            while curr:
                ret.append(curr.pos)
                curr = curr.parent
            return ret[::-1]

        neighbors = maze.getNeighbors(curr.pos[0], curr.pos[1])

        for neighbor in neighbors:
            if neighbor not in closed_list:
                tmp = starNode(neighbor)
                tmp.parent = curr
                tmp.g = curr.g + 1
                tmp.h = manhattan_distance(tmp.pos, end_node.pos)
                tmp.f = tmp.g + tmp.h
                heapq.heappush(open_list, (tmp.f, tmp.pos, tmp))
                closed_list.add(tmp.pos)

    return []

def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start, objectives = maze.getStart(), set(maze.getObjectives())
    open_list, ret, counter = [], [], 0
    mst_cache = dict()
    graph = {i: {j: len(customAStar(maze, i, j)) for j in objectives if j != i} for i in objectives}
    all_targets_key = str(sorted(objectives))
    total_mst = mst_cache.setdefault(all_targets_key, kruskalMST(graph, objectives))

    for objective in objectives:
        node = starNodeMultiple(start)
        node.targets = copy.deepcopy(objectives)
        node.currentTarget = objective
        node.seen.add(node.pos)
        node.g = 0
        node.h = total_mst + manhattan_distance(node.pos, node.currentTarget)
        node.f = node.g + node.h
        heapq.heappush(open_list, (node.f, counter, node))
        counter += 1

    while open_list:
        curr = heapq.heappop(open_list)[2]

        if curr.pos in curr.targets and curr.pos != curr.currentTarget:
            continue

        if curr.pos == curr.currentTarget:
            curr.targets.remove(curr.pos)

            if not curr.targets:
                while curr:
                    ret.append(curr.pos)
                    curr = curr.parent
                return ret[::-1]

            curr.seen.clear()
            curr.seen.add(curr.pos)

            for target in curr.targets:
                node = starNodeMultiple(curr.pos)
                node.targets = copy.deepcopy(curr.targets)
                node.currentTarget = target
                node.seen = copy.deepcopy(curr.seen)
                node.parent = curr.parent
                node.g = curr.g
                remTargets = str(sorted(node.targets))
                remMST = mst_cache.setdefault(remTargets, kruskalMST(graph, node.targets))
                node.h = remMST + manhattan_distance(node.pos, node.currentTarget)
                node.f = node.g + node.h
                heapq.heappush(open_list, (node.f, counter, node))
                counter += 1
            continue

        neighbors = maze.getNeighbors(curr.pos[0], curr.pos[1])

        for neighbor in neighbors:
            if neighbor not in curr.seen:
                node = starNodeMultiple(neighbor)
                node.parent = curr
                node.targets = copy.deepcopy(curr.targets)
                node.currentTarget = curr.currentTarget
                node.seen = curr.seen
                node.seen.add(node.pos)
                node.g = curr.g + 1
                remTargets = str(sorted(node.targets))
                node.h = mst_cache[remTargets] + manhattan_distance(node.pos, node.currentTarget)
                node.f = node.g + node.h
                heapq.heappush(open_list, (node.f, counter, node))
                counter += 1

    return []

#Function that returns Manhattan distance from current pos to closest remaining target
def distance_to_nearest_objective(currPos, targets):
    reto = float('inf')
    for target in targets:
        reto = min(reto, manhattan_distance(currPos, target))
    return reto

def nearest_objective(currPos, targets):
    ret = (float('inf'), float('inf'))
    for target in targets:
        if manhattan_distance(currPos, target) < manhattan_distance(currPos, ret):
            ret = target
    return ret

def manhattan_distance(currPos, targetPos):
    return abs(currPos[0] - targetPos[0]) + abs(currPos[1] - targetPos[1])

def distance_to_remaining_objectives(currPos, targets):
    ret = 0
    for target in targets:
        ret += manhattan_distance(currPos, target)
    return ret

#My own kruskal prep function that takes in graph and targets set and creates edges list and vertex list to pass into kruskal function
def kruskalMST(graphWeights, targets):
    edges = []
    verts = [i for i in graphWeights.keys() if i in targets]
    for vert in verts:
        for key2, val in graphWeights[vert].items():
            if key2 in targets:
                edges.append((val, vert, key2))
    currMST = kruskals(verts, edges)
    ret = 0
    for i in range(len(currMST)):
        ret += currMST[i][0]
    return ret

def mst(graph, targets):
    vals, largest = 0, float('-inf')

    if len(targets) == 1:
        return 0

    for target in targets:
        tmp = float('inf')
        for key, val in graph[target].items():
            timp = min(tmp, val)
        largest = max(largest, tmp)
        vals += tmp
    return vals - largest

def customAStar(maze, start, end):
    start_node, end_node = starNode(start), starNode(end)
    start_node.g = start_node.h = start_node.f = end_node.g = end_node.h = end_node.f = 0
    open_list, closed_list = [], set()
    heapq.heappush(open_list, (start_node.f, start_node.pos, start_node))
    ret = []
    closed_list.add(start_node.pos)

    while open_list:
        curr = heapq.heappop(open_list)[2]

        if curr.pos == end_node.pos:
            while curr:
                ret.append(curr.pos)
                curr = curr.parent
            return ret[::-1]

        neighbors = maze.getNeighbors(curr.pos[0], curr.pos[1])

        for neighbor in neighbors:
            if neighbor not in closed_list:
                node = starNode(neighbor)
                node.parent = curr
                node.g = curr.g + 1
                node.h = manhattan_distance(node.pos, end_node.pos)
                node.f = node.g + node.h
                heapq.heappush(open_list, (node.f, node.pos, node))
                closed_list.add(node.pos)

    return []

"""
These two dictionaries are part of referenced kruskal algorithm for the disjoint sets implementation. The functions kruskals(), make_set(), find(), and union() are all referenced from
https://gist.github.com/hayderimran7/09960ca438a65a9bd10d0254b792f48f and are slightly altered in their data types to fit my input and requirements. Furthermore, they are ONLY used for my heuristic and nothing more.
"""
parent = dict()
rank = dict()

def kruskals(graph, edgess):
    for vertice in graph:
        make_set(vertice)
        minimum_spanning_tree = set()
        edges = list(edgess)
        edges.sort()
        #print edges
    for edge in edges:
        weight, vertice1, vertice2 = edge
        if find(vertice1) != find(vertice2):
            union(vertice1, vertice2)
            minimum_spanning_tree.add(edge)

    return sorted(minimum_spanning_tree)

def make_set(vertice):
    parent[vertice] = vertice
    rank[vertice] = 0

def find(vertice):
    if parent[vertice] != vertice:
        parent[vertice] = find(parent[vertice])
    return parent[vertice]

def union(vertice1, vertice2):
    root1 = find(vertice1)
    root2 = find(vertice2)
    if root1 != root2:
        if rank[root1] > rank[root2]:
            parent[root2] = root1
        else:
	        parent[root1] = root2
        if rank[root1] == rank[root2]: rank[root2] += 1


def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return astar_corner(maze)

def extra(maze):
    """
    Runs extra credit suggestion.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start, objectives = maze.getStart(), set(maze.getObjectives())
    open_list, ret, counter = [], [], 0
    mst_cache = dict()
    graph = {i: {j: manhattan_distance(i, j) for j in objectives if j != i} for i in objectives}
    allTargets = str(sorted(objectives))
    total_mst = mst_cache.setdefault(allTargets, mst(graph, objectives))

    for objective in objectives:
        node = starNodeMultiple(start)
        node.targets = copy.deepcopy(objectives)
        node.currentTarget = objective
        node.seen.add(node.pos)
        node.g = 0
        node.h = manhattan_distance(node.pos, node.currentTarget) + len(node.targets) + total_mst
        node.f = node.g + node.h
        heapq.heappush(open_list, (node.f, counter, node))
        counter += 1

    while open_list:
        curr = heapq.heappop(open_list)[2]

        if curr.pos == curr.currentTarget:
            curr.targets.remove(curr.pos)
            print(len(curr.targets))

            if not curr.targets:
                while curr:
                    ret.append(curr.pos)
                    curr = curr.parent
                return ret[::-1]

            curr.seen.clear()
            curr.seen.add(curr.pos)

            for target in curr.targets:
                node = starNodeMultiple(curr.pos)
                node.targets = copy.deepcopy(curr.targets)
                node.currentTarget = target
                node.seen = copy.deepcopy(curr.seen)
                node.parent = curr.parent
                node.g = curr.g
                remTargets = str(sorted(node.targets))
                remMST = mst_cache.setdefault(remTargets, mst(graph, node.targets))
                node.h = manhattan_distance(node.pos, node.currentTarget) + len(node.targets) + remMST
                node.f = node.g + node.h
                heapq.heappush(open_list, (node.f, counter, node))
                counter += 1
            continue

        neighbors = maze.getNeighbors(curr.pos[0], curr.pos[1])

        for neighbor in neighbors:
            if neighbor not in curr.seen:
                node = starNodeMultiple(neighbor)
                node.parent = curr
                node.targets = copy.deepcopy(curr.targets)
                node.currentTarget = curr.currentTarget
                node.seen = curr.seen
                node.seen.add(node.pos)
                node.g = curr.g + 1
                remTargets = str(sorted(node.targets))
                node.h = mst_cache[remTargets] + manhattan_distance(node.pos, node.currentTarget) + len(node.targets)
                node.f = node.g + node.h
                heapq.heappush(open_list, (node.f, counter, node))
                counter += 1

    return []

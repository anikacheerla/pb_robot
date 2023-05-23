from collections import namedtuple, Mapping
from heapq import heappop, heappush
import numpy as np
import operator
import time

import copy

from .util import INF, merge_dicts, flatten, bisect

import pb_robot

# TODO - Visibility-PRM, PRM*

class Vertex(object):
    def __init__(self, q):
        self.q = q
        self.edges = {}
        self._handle = None

    def clear(self):
        self._handle = None

    def __str__(self):
        return 'Vertex(' + str(self.q) + ')'
    __repr__ = __str__
    
class Edge(object):
    def __init__(self, v1, v2, path):
        self.v1, self.v2 = v1, v2
        self.v1.edges[v2], self.v2.edges[v1] = self, self
        self._path = path
        #self._handle = None
        self._handles = []

    def end(self, start):
        if self.v1 == start:
            return self.v2
        if self.v2 == start:
            return self.v1
        assert False

    def path(self, start, t1=0, t2=0):
        if self._path is None:
            return [self.end(start).q]
        if self.v1 == start:
            return self._path + [self.v2.q]
        if self.v2 == start:
            return self._path[::-1] + [self.v1.q]
        assert False

    def configs(self):
        if self._path is None:
            return []
        return [self.v1.q] + self._path + [self.v2.q]

    def clear(self):
        #self._handle = None
        self._handles = []

    def __str__(self):
        return 'Edge(' + str(self.v1.q) + ' - ' + str(self.v2.q) + ')'
    __repr__ = __str__

class TimeVertex(Vertex):
    def __init__(self, q):
        super(self.__class__, self).__init__(q)
        self.times = [] # times the vertex is available
    
    # def __hash__(self):
    #     return hash(self.q.tostring())
    
    def is_active(self, t):
        # TODO: can be much much smarter about this, change to binary search, MRU ordering
        for tup in self.times:
            if t < tup[1] and t >= tup[0]: # in a collision time
                return False, tup[1]
        return True, 0

class TimeEdge(Edge):
    def __init__(self, v1, v2, path):
        super(self.__class__, self).__init__(v1, v2, path)
        self.times = [] # times the edge is available
    
    def path(self, start, t1=0, t2=0):
        if self._path is None:
            return [(self.end(start).q, t2)]
        if self.v1 == start:
            return [(q, t) for q, t in zip(self._path, np.linspace(t1, t2, len(self._path)))] + [(self.v2.q, t2)]
        if self.v2 == start:
            return [(q, t) for q, t in zip(self._path[::-1], np.linspace(t1, t2, len(self._path)))] + [(self.v1.q, t2)]
        assert False

    def is_edge_active(self, t, duration=0):
        # TODO: can be much much smarter about this, change to binary search, MRU ordering
        # make sure that t -> duration isn't overlapping with 
        for tup in self.times:
            if t <= tup[1] and t >= tup[0]: # in a collision time
                return False, tup[1] # when edge will next be free
        return True, 0

##################################################

SearchNode = namedtuple('SearchNode', ['cost', 'parent'])

class Roadmap(Mapping, object):
    def __init__(self, samples=[]):
        self.vertices = {}
        self.edges = []
        self.add(samples)

    def __getitem__(self, q):
        return self.vertices[q.tostring()]
    
    def __setitem__(self, q, value):
        self.vertices[q.tostring()] = value

    def __len__(self):
        return len(self.vertices)

    def __iter__(self):
        return iter(self.vertices)

    def __call__(self, q1, q2):
        if q1 not in self or q2 not in self:
            return None
        start, goal = self[q1], self[q2]
        queue = [(0, start)]
        nodes, processed = {start: SearchNode(0, None)}, set()

        def retrace(v):
            pv = nodes[v].parent
            if pv is None:
                return [v.q]
            return retrace(pv) + v.edges[pv].path(pv)

        while len(queue) != 0:
            _, cv = heappop(queue)
            if cv in processed:
                continue
            processed.add(cv)
            if cv == goal:
                return retrace(cv)
            for nv, edge in cv.edges.items():
                cost = nodes[cv].cost + len(edge.path(cv))
                if nv not in nodes or cost < nodes[nv].cost:
                    nodes[nv] = SearchNode(cost, cv)
                    heappush(queue, (cost, nv))
        return None

    def add(self, samples):
        new_vertices = []
        for q in samples:
            if q not in self:
                self[q] = Vertex(q)
                new_vertices.append(self[q])
        return new_vertices

    def connect(self, v1, v2, path=None):
        if v1 not in v2.edges:  # TODO - what about parallel edges?
            edge = Edge(v1, v2, path)
            self.edges.append(edge)
            return edge
        return None

    def clear(self):
        for v in self.vertices.values():
            v.clear()
        for e in self.edges:
            e.clear()

    def draw(self, env):
        for v in self.vertices.values():
            v.draw(env)
        return 0

        for e in self.edges:
            e.draw(env)

    @staticmethod
    def merge(*roadmaps):
        new_roadmap = Roadmap()
        new_roadmap.vertices = merge_dicts(
            *[roadmap.vertices for roadmap in roadmaps])
        new_roadmap.edges = list(
            flatten(roadmap.edges for roadmap in roadmaps))
        return new_roadmap


class PRM(Roadmap):
    def __init__(self, distance_fn, extend_fn, collision_fn, samples=[]):
        super(PRM, self).__init__()
        self.distance_fn = distance_fn
        self.extend_fn = extend_fn
        self.collision_fn = collision_fn

    def grow(self, samples):
        raise NotImplementedError()

    def __call__(self, q1, q2):
        self.grow(samples=[q1, q2])
        if q1 not in self or q2 not in self:
            return None
        start, goal = self[q1], self[q2]
        heuristic = lambda v: self.distance_fn(v.q, goal.q)  # lambda v: 0

        queue = [(heuristic(start), start)]
        nodes, processed = {start: SearchNode(0, None)}, set()

        def retrace(v):
            if nodes[v].parent is None:
                return [v.q]
            return retrace(nodes[v].parent) + v.edges[nodes[v].parent].path(nodes[v].parent)

        while len(queue) != 0:
            _, cv = heappop(queue)
            if cv in processed:
                continue
            processed.add(cv)
            if cv == goal:
                return retrace(cv)
            for nv in cv.edges:
                cost = nodes[cv].cost + self.distance_fn(cv.q, nv.q)
                if (nv not in nodes) or (cost < nodes[nv].cost):
                    nodes[nv] = SearchNode(cost, cv)
                    heappush(queue, (cost + heuristic(nv), nv))
        return None

##################################################

class TPRM(PRM):
    def __init__(self, manip, distance_fn, extend_fn, collision_fn, static_obstacles, 
                 dynamic_obstacles, edge_collision_fn = None, samples=[], connect_distance=2.5, json_file="graph.json"):
        self.manip = manip
        self.connect_distance = connect_distance # should be less than the dimensions of an obstacles
        self.static_obstacles = static_obstacles
        self.dynamic_obstacles = dynamic_obstacles
        self.edge_collision_fn = edge_collision_fn

        super(self.__class__, self).__init__(
            distance_fn, extend_fn, collision_fn, samples=samples)
        
        self.json_file = json_file
        # self.grow(samples)
        if json_file:
            self.load_graph(json_file)
        else:
            self.grow(samples)

    
    def load_graph(self, json_file):
        import json
        with open(json_file, 'r') as openfile:
            # Reading from json file
            json_object = json.load(openfile)
            

            for i, val in json_object.items():
                v1 = TimeVertex(np.array(val[0]))
                v2 = TimeVertex(np.array(val[1]))
                path = val[2]
                times = val[3]
                edge = TimeEdge(v1, v2, path)
                edge.times = times
                self.edges.append(edge) 

    def save_graph(self, json_file):
        import json
        graph = dict()

        # Save each edge: q1, q2: path, times
        # initialize vertices by looping through edges, intialize times

        for i, e in enumerate(self.edges): 
            graph[i] = (e.v1.q.tolist(), e.v2.q.tolist(), e._path, e.times)
        with open(json_file, 'w') as outfile:
            json.dump(graph, outfile)
                
    def duration(self, v1, v2):
        # Calculate time taken to get from q1 to q2
        # Everything takes 1 sec per unti distance for now
        if v1 == v2:
            return 1
        return self.distance_fn(v1.q, v2.q) # * robot movement speed
    
    def calculate_time_availabilities(self, vertices, dynamic_obstacles, timestep=0.5, max_time=1.5):
        if not dynamic_obstacles:
            return

        collision_start = [-1]*len(vertices)
        for t in np.arange(0, max_time, timestep):
            for obs in dynamic_obstacles: 
                obs.set_point(obs.get_position(t))

            for i, v in enumerate(vertices):
                # in collision
                if collision_start[i] == -1 and not self.manip.IsCollisionFree(v.q, obstacles=dynamic_obstacles):
                    collision_start[i] = t
                # out of collision
                if collision_start[i] != -1 and self.manip.IsCollisionFree(v.q, obstacles=dynamic_obstacles):
                    v.times.append((collision_start[i], t))
                    collision_start[i] = -1
            
        for obs in dynamic_obstacles:
            obs.set_point(obs.get_position(0))


        for i, v in enumerate(vertices):
            if collision_start[i] != -1:
                v.times.append((collision_start[i], max_time))
    
    def calculate_edge_time_availabilities(self, edges, dynamic_obstacles, timestep=0.5, max_time=1.5):
        if not dynamic_obstacles:
            return 
        
        collision_start = [-1]*len(edges)
        for t in np.arange(0, max_time, timestep):
            for obs in dynamic_obstacles:
                obs.set_point(obs.get_position(t))
            for i, e in enumerate(edges):
                if collision_start[i] == -1:
                    # in collision
                    if any (not self.manip.IsCollisionFree(q, obstacles=dynamic_obstacles) for q in bisect(e._path)):
                        collision_start[i] = t
                elif all (self.manip.IsCollisionFree(q, obstacles=dynamic_obstacles) for q in bisect(e._path)):
                    # now not collision free
                    e.times.append((collision_start[i], t))
                    collision_start[i] = -1

        for obs in dynamic_obstacles:
            obs.set_point(obs.get_position(0))

        for i, e in enumerate(edges):
            if collision_start[i] != -1:
                e.times.append((collision_start[i], max_time))


    def __call__(self, q1, q2, start_time=0):
        # TODO: start_time is global time at which previous path ended
        self.grow(samples=[q1, q2])

        if q1 not in self or q2 not in self:
            return None
        start, goal = self[q1], self[q2]
        heuristic = lambda v: self.distance_fn(v.q, goal.q)  # lambda v: 0

        queue = [(heuristic(start), start)]
        nodes, processed = {start: SearchNode(start_time, None)}, set()

        arrival_times = dict()
        arrival_times[start] = start_time

        def retrace(v):
            if nodes[v].parent is None:
                return [v.q]
            return retrace(nodes[v].parent) + v.edges[nodes[v].parent].path(nodes[v].parent)
        
        def retrace_with_times(v):
            # print(arrival_times[v])
            if nodes[v].parent is None:
                assert arrival_times[v] == start_time
                return [(v.q, arrival_times[v])]
            parent = nodes[v].parent
            return retrace_with_times(parent) + v.edges[parent].path(parent, arrival_times[parent], arrival_times[v])
        
        while len(queue) != 0:
            _, cv = heappop(queue)
            t = arrival_times[cv]
            if cv in processed:
                continue
            processed.add(cv)
            if cv == goal:
                return retrace_with_times(cv), t # return end_time as well
            for nv, edge in cv.edges.items():
                duration = self.duration(cv, nv)
                is_active, next_free = edge.is_edge_active(t+duration)
                if is_active:
                    # print("Time:" ,t+duration)

                    cost = nodes[cv].cost + self.distance_fn(cv.q, nv.q)
                    if (nv not in nodes) or (cost < nodes[nv].cost):
                        nodes[nv] = SearchNode(cost, cv)
                        arrival_times[nv] = t + duration
                        if cv != nv:
                            heappush(queue, (cost + heuristic(nv), nv))

                else: # does it help to wait until that edge is active?
                    wait_time = next_free - (t + duration) # find next free interval of node
                    assert nv.is_active(next_free)[0]
                    cost = nodes[cv].cost + self.distance_fn(cv.q, nv.q) + 0.01*wait_time # function of wait time gives wait penalty
                    if (nv not in nodes) or (cost < nodes[nv].cost):
                        nodes[nv] = SearchNode(cost, cv)
                        arrival_times[nv] = next_free
                        if cv != nv:
                            heappush(queue, (cost + heuristic(nv), nv))
        print("Plan motion FAILED")
        # import IPython
        # IPython.embed()
        return None, None
    
    def connect(self, v1, v2, path=None):
        if v1 not in v2.edges:  # TODO - what about parallel edges?
            edge = TimeEdge(v1, v2, path)
            self.edges.append(edge)
            return edge
        return None

    def grow(self, samples):
        old_vertices = self.vertices.values()
        new_vertices = self.add(samples)

        new_edges = []

        for i, v1 in enumerate(new_vertices):
            for v2 in new_vertices[i+1:] + list(old_vertices):
                if (v1 == v2):
                    # TODO: why is this happening??
                    continue
                if self.distance_fn(v1.q, v2.q) <= self.connect_distance:
                    path = list(self.extend_fn(v1.q, v2.q))[:-1]
                    if not any(self.collision_fn(q) for q in bisect(path)):
                        edge = self.connect(v1, v2, path)
                        if edge:
                            new_edges.append(edge)
        
        self.calculate_edge_time_availabilities(new_edges, self.dynamic_obstacles)
        if not self.json_file:
            self.save_graph("graph_hard.json")
        return new_vertices

    def add(self, samples):
        new_vertices = []

        for q in samples:
            if q not in self:
                self[q] = TimeVertex(q) # should initialize with time avails
                new_vertices.append(self[q])
        
        # self.calculate_time_availabilities(new_vertices, self.dynamic_obstacles)
        # for v in new_vertices:
        #     if len(v.times) > 0:
        #         print(v.times)

        return new_vertices
import math
from functools import lru_cache
import numpy as np
from multiprocessing import Pool, cpu_count

CACHE_SIZE = 1000
PARALLEL_TOLERANCE = 1e-16


class Point:
    def __init__(self, x, y, z):
        self._x = x
        self._y = y
        self._z = z
        self._hashcode = hash((x, y, z))

    def get_coordinates(self):
        return self._x, self._y, self._z

    def __str__(self):
        return str(self.get_coordinates())

    def x(self):
        return self._x

    def y(self):
        return self._y

    def z(self):
        return self._z

    def __hash__(self):
        return self._hashcode


@lru_cache(maxsize=CACHE_SIZE)
def distance_between_points(point1, point2):
    coordinates1 = np.asarray(point1.get_coordinates())
    coordinates2 = np.asarray(point2.get_coordinates())
    squared_distance = np.sum((coordinates1 - coordinates2) ** 2)
    distance = math.sqrt(squared_distance)
    return distance


class Edge:
    def __init__(self, point1, point2):
        self._point1 = point1
        self._point2 = point2
        self._hash = hash((point1, point2))

    def get_points(self):
        return [self._point1, self._point2]

    def __hash__(self):
        return self._hash


class Face:
    def __init__(self, point1, point2, point3):
        self._point1 = point1
        self._point2 = point2
        self._point3 = point3
        self._hashcode = hash((point1, point2, point3))

    def get_points(self):
        return [self._point1, self._point2, self._point3]

    def get_edges(self):
        return [Edge(self._point1, self._point2), Edge(self._point1, self._point3), Edge(self._point2, self._point3)]

    def __hash__(self):
        return self._hashcode


@lru_cache(maxsize=CACHE_SIZE)
def distance_between_face_and_point(face, point):
    alpha, beta, gamma, delta = compute_plane_equation(face)
    x, y, z = point.get_coordinates()
    c = -(alpha * x + beta * y + delta + gamma * z) / (alpha ** 2 + beta ** 2 + gamma ** 2)
    nearest_point = Point(x + alpha * c, y + beta * c, z + gamma * c)

    if point_belongs_to_face(nearest_point, face):
        return distance_between_points(nearest_point, point)
    else:
        vertex_a, vertex_b, vertex_c = face.get_points()
        edges = [Edge(vertex_a, vertex_b), Edge(vertex_a, vertex_c), Edge(vertex_b, vertex_c)]
        minimal_distance = math.inf
        for edge in edges:
            distance = distance_between_edge_and_point(edge, point)
            if distance < minimal_distance:
                minimal_distance = distance
        return minimal_distance


@lru_cache(maxsize=CACHE_SIZE)
def point_belongs_to_face(point, face):
    px, py, pz = point.get_coordinates()
    ax, ay, az = face.get_points()[0].get_coordinates()
    bx, by, bz = face.get_points()[1].get_coordinates()
    cx, cy, cz = face.get_points()[2].get_coordinates()

    x = math.nan
    y = math.nan
    if ((ax - cx) * (by - cy) - (ay - cy) * (bx - cx)) != 0 and ((ax - cx) * (by - cy) - (ay - cy) * (bx - cx)) != 0:
        x = ((bx - cx) * (cy - py) - (by - cy) * (cx - px)) / ((ax - cx) * (by - cy) - (ay - cy) * (bx - cx))
        y = (-(ax - cx) * (cy - py) + (ay - cy) * (cx - px)) / ((ax - cx) * (by - cy) - (ay - cy) * (bx - cx))
    elif (math.isnan(x) or math.isnan(y)) and ((ax - cx) * (bz - cz) - (az - cz) * (bx - cx)) != 0 and (
            (ax - cx) * (bz - cz) - (az - cz) * (bx - cx)) != 0:
        x = ((bx - cx) * (cz - pz) - (bz - cz) * (cx - px)) / ((ax - cx) * (bz - cz) - (az - cz) * (bx - cx))
        y = (-(ax - cx) * (cz - pz) + (az - cz) * (cx - px)) / ((ax - cx) * (bz - cz) - (az - cz) * (bx - cx))
    elif (math.isnan(x) or math.isnan(y)) and ((ay - cy) * (bz - cz) - (az - cz) * (by - cy)) != 0 and (
            (ay - cy) * (bz - cz) - (az - cz) * (by - cy)) != 0:
        x = ((by - cy) * (cz - pz) - (bz - cz) * (cy - py)) / ((ay - cy) * (bz - cz) - (az - cz) * (by - cy))
        y = (-(ay - cy) * (cz - pz) + (az - cz) * (cy - py)) / ((ay - cy) * (bz - cz) - (az - cz) * (by - cy))

    return not math.isnan(x) and not math.isnan(y) and x >= 0 and y >= 0 and x + y <= 1


@lru_cache(maxsize=CACHE_SIZE)
def compute_plane_equation(face):
    x1, y1, z1 = face.get_points()[0].get_coordinates()
    x2, y2, z2 = face.get_points()[1].get_coordinates()
    x3, y3, z3 = face.get_points()[2].get_coordinates()

    alpha = (-y1 * z2 + y1 * z3 + y2 * z1 - y2 * z3 - y3 * z1 + y3 * z2) * math.sqrt(1 / (
        x1 ** 2 * y2 ** 2 - 2 * x1 ** 2 * y2 * y3 + x1 ** 2 * y3 ** 2 + x1 ** 2 * z2 ** 2 - 2 * x1 ** 2 * z2 * z3 + x1 ** 2 * z3 ** 2 - 2 * x1 * x2 * y1 * y2 + 2 * x1 * x2 * y1 * y3 + 2 * x1 * x2 * y2 * y3 - 2 * x1 * x2 * y3 ** 2 - 2 * x1 * x2 * z1 * z2 + 2 * x1 * x2 * z1 * z3 + 2 * x1 * x2 * z2 * z3 - 2 * x1 * x2 * z3 ** 2 + 2 * x1 * x3 * y1 * y2 - 2 * x1 * x3 * y1 * y3 - 2 * x1 * x3 * y2 ** 2 + 2 * x1 * x3 * y2 * y3 + 2 * x1 * x3 * z1 * z2 - 2 * x1 * x3 * z1 * z3 - 2 * x1 * x3 * z2 ** 2 + 2 * x1 * x3 * z2 * z3 + x2 ** 2 * y1 ** 2 - 2 * x2 ** 2 * y1 * y3 + x2 ** 2 * y3 ** 2 + x2 ** 2 * z1 ** 2 - 2 * x2 ** 2 * z1 * z3 + x2 ** 2 * z3 ** 2 - 2 * x2 * x3 * y1 ** 2 + 2 * x2 * x3 * y1 * y2 + 2 * x2 * x3 * y1 * y3 - 2 * x2 * x3 * y2 * y3 - 2 * x2 * x3 * z1 ** 2 + 2 * x2 * x3 * z1 * z2 + 2 * x2 * x3 * z1 * z3 - 2 * x2 * x3 * z2 * z3 + x3 ** 2 * y1 ** 2 - 2 * x3 ** 2 * y1 * y2 + x3 ** 2 * y2 ** 2 + x3 ** 2 * z1 ** 2 - 2 * x3 ** 2 * z1 * z2 + x3 ** 2 * z2 ** 2 + y1 ** 2 * z2 ** 2 - 2 * y1 ** 2 * z2 * z3 + y1 ** 2 * z3 ** 2 - 2 * y1 * y2 * z1 * z2 + 2 * y1 * y2 * z1 * z3 + 2 * y1 * y2 * z2 * z3 - 2 * y1 * y2 * z3 ** 2 + 2 * y1 * y3 * z1 * z2 - 2 * y1 * y3 * z1 * z3 - 2 * y1 * y3 * z2 ** 2 + 2 * y1 * y3 * z2 * z3 + y2 ** 2 * z1 ** 2 - 2 * y2 ** 2 * z1 * z3 + y2 ** 2 * z3 ** 2 - 2 * y2 * y3 * z1 ** 2 + 2 * y2 * y3 * z1 * z2 + 2 * y2 * y3 * z1 * z3 - 2 * y2 * y3 * z2 * z3 + y3 ** 2 * z1 ** 2 - 2 * y3 ** 2 * z1 * z2 + y3 ** 2 * z2 ** 2))

    beta = (x1 * z2 - x1 * z3 - x2 * z1 + x2 * z3 + x3 * z1 - x3 * z2) * math.sqrt(1 / (
        x1 ** 2 * y2 ** 2 - 2 * x1 ** 2 * y2 * y3 + x1 ** 2 * y3 ** 2 + x1 ** 2 * z2 ** 2 - 2 * x1 ** 2 * z2 * z3 + x1 ** 2 * z3 ** 2 - 2 * x1 * x2 * y1 * y2 + 2 * x1 * x2 * y1 * y3 + 2 * x1 * x2 * y2 * y3 - 2 * x1 * x2 * y3 ** 2 - 2 * x1 * x2 * z1 * z2 + 2 * x1 * x2 * z1 * z3 + 2 * x1 * x2 * z2 * z3 - 2 * x1 * x2 * z3 ** 2 + 2 * x1 * x3 * y1 * y2 - 2 * x1 * x3 * y1 * y3 - 2 * x1 * x3 * y2 ** 2 + 2 * x1 * x3 * y2 * y3 + 2 * x1 * x3 * z1 * z2 - 2 * x1 * x3 * z1 * z3 - 2 * x1 * x3 * z2 ** 2 + 2 * x1 * x3 * z2 * z3 + x2 ** 2 * y1 ** 2 - 2 * x2 ** 2 * y1 * y3 + x2 ** 2 * y3 ** 2 + x2 ** 2 * z1 ** 2 - 2 * x2 ** 2 * z1 * z3 + x2 ** 2 * z3 ** 2 - 2 * x2 * x3 * y1 ** 2 + 2 * x2 * x3 * y1 * y2 + 2 * x2 * x3 * y1 * y3 - 2 * x2 * x3 * y2 * y3 - 2 * x2 * x3 * z1 ** 2 + 2 * x2 * x3 * z1 * z2 + 2 * x2 * x3 * z1 * z3 - 2 * x2 * x3 * z2 * z3 + x3 ** 2 * y1 ** 2 - 2 * x3 ** 2 * y1 * y2 + x3 ** 2 * y2 ** 2 + x3 ** 2 * z1 ** 2 - 2 * x3 ** 2 * z1 * z2 + x3 ** 2 * z2 ** 2 + y1 ** 2 * z2 ** 2 - 2 * y1 ** 2 * z2 * z3 + y1 ** 2 * z3 ** 2 - 2 * y1 * y2 * z1 * z2 + 2 * y1 * y2 * z1 * z3 + 2 * y1 * y2 * z2 * z3 - 2 * y1 * y2 * z3 ** 2 + 2 * y1 * y3 * z1 * z2 - 2 * y1 * y3 * z1 * z3 - 2 * y1 * y3 * z2 ** 2 + 2 * y1 * y3 * z2 * z3 + y2 ** 2 * z1 ** 2 - 2 * y2 ** 2 * z1 * z3 + y2 ** 2 * z3 ** 2 - 2 * y2 * y3 * z1 ** 2 + 2 * y2 * y3 * z1 * z2 + 2 * y2 * y3 * z1 * z3 - 2 * y2 * y3 * z2 * z3 + y3 ** 2 * z1 ** 2 - 2 * y3 ** 2 * z1 * z2 + y3 ** 2 * z2 ** 2))

    gamma = (-x1 * y2 + x1 * y3 + x2 * y1 - x2 * y3 - x3 * y1 + x3 * y2) * math.sqrt(1 / (
        x1 ** 2 * y2 ** 2 - 2 * x1 ** 2 * y2 * y3 + x1 ** 2 * y3 ** 2 + x1 ** 2 * z2 ** 2 - 2 * x1 ** 2 * z2 * z3 + x1 ** 2 * z3 ** 2 - 2 * x1 * x2 * y1 * y2 + 2 * x1 * x2 * y1 * y3 + 2 * x1 * x2 * y2 * y3 - 2 * x1 * x2 * y3 ** 2 - 2 * x1 * x2 * z1 * z2 + 2 * x1 * x2 * z1 * z3 + 2 * x1 * x2 * z2 * z3 - 2 * x1 * x2 * z3 ** 2 + 2 * x1 * x3 * y1 * y2 - 2 * x1 * x3 * y1 * y3 - 2 * x1 * x3 * y2 ** 2 + 2 * x1 * x3 * y2 * y3 + 2 * x1 * x3 * z1 * z2 - 2 * x1 * x3 * z1 * z3 - 2 * x1 * x3 * z2 ** 2 + 2 * x1 * x3 * z2 * z3 + x2 ** 2 * y1 ** 2 - 2 * x2 ** 2 * y1 * y3 + x2 ** 2 * y3 ** 2 + x2 ** 2 * z1 ** 2 - 2 * x2 ** 2 * z1 * z3 + x2 ** 2 * z3 ** 2 - 2 * x2 * x3 * y1 ** 2 + 2 * x2 * x3 * y1 * y2 + 2 * x2 * x3 * y1 * y3 - 2 * x2 * x3 * y2 * y3 - 2 * x2 * x3 * z1 ** 2 + 2 * x2 * x3 * z1 * z2 + 2 * x2 * x3 * z1 * z3 - 2 * x2 * x3 * z2 * z3 + x3 ** 2 * y1 ** 2 - 2 * x3 ** 2 * y1 * y2 + x3 ** 2 * y2 ** 2 + x3 ** 2 * z1 ** 2 - 2 * x3 ** 2 * z1 * z2 + x3 ** 2 * z2 ** 2 + y1 ** 2 * z2 ** 2 - 2 * y1 ** 2 * z2 * z3 + y1 ** 2 * z3 ** 2 - 2 * y1 * y2 * z1 * z2 + 2 * y1 * y2 * z1 * z3 + 2 * y1 * y2 * z2 * z3 - 2 * y1 * y2 * z3 ** 2 + 2 * y1 * y3 * z1 * z2 - 2 * y1 * y3 * z1 * z3 - 2 * y1 * y3 * z2 ** 2 + 2 * y1 * y3 * z2 * z3 + y2 ** 2 * z1 ** 2 - 2 * y2 ** 2 * z1 * z3 + y2 ** 2 * z3 ** 2 - 2 * y2 * y3 * z1 ** 2 + 2 * y2 * y3 * z1 * z2 + 2 * y2 * y3 * z1 * z3 - 2 * y2 * y3 * z2 * z3 + y3 ** 2 * z1 ** 2 - 2 * y3 ** 2 * z1 * z2 + y3 ** 2 * z2 ** 2))

    delta = (x1 * y2 * z3 - x1 * y3 * z2 - x2 * y1 * z3 + x2 * y3 * z1 + x3 * y1 * z2 - x3 * y2 * z1) * math.sqrt(1 / (
        x1 ** 2 * y2 ** 2 - 2 * x1 ** 2 * y2 * y3 + x1 ** 2 * y3 ** 2 + x1 ** 2 * z2 ** 2 - 2 * x1 ** 2 * z2 * z3 + x1 ** 2 * z3 ** 2 - 2 * x1 * x2 * y1 * y2 + 2 * x1 * x2 * y1 * y3 + 2 * x1 * x2 * y2 * y3 - 2 * x1 * x2 * y3 ** 2 - 2 * x1 * x2 * z1 * z2 + 2 * x1 * x2 * z1 * z3 + 2 * x1 * x2 * z2 * z3 - 2 * x1 * x2 * z3 ** 2 + 2 * x1 * x3 * y1 * y2 - 2 * x1 * x3 * y1 * y3 - 2 * x1 * x3 * y2 ** 2 + 2 * x1 * x3 * y2 * y3 + 2 * x1 * x3 * z1 * z2 - 2 * x1 * x3 * z1 * z3 - 2 * x1 * x3 * z2 ** 2 + 2 * x1 * x3 * z2 * z3 + x2 ** 2 * y1 ** 2 - 2 * x2 ** 2 * y1 * y3 + x2 ** 2 * y3 ** 2 + x2 ** 2 * z1 ** 2 - 2 * x2 ** 2 * z1 * z3 + x2 ** 2 * z3 ** 2 - 2 * x2 * x3 * y1 ** 2 + 2 * x2 * x3 * y1 * y2 + 2 * x2 * x3 * y1 * y3 - 2 * x2 * x3 * y2 * y3 - 2 * x2 * x3 * z1 ** 2 + 2 * x2 * x3 * z1 * z2 + 2 * x2 * x3 * z1 * z3 - 2 * x2 * x3 * z2 * z3 + x3 ** 2 * y1 ** 2 - 2 * x3 ** 2 * y1 * y2 + x3 ** 2 * y2 ** 2 + x3 ** 2 * z1 ** 2 - 2 * x3 ** 2 * z1 * z2 + x3 ** 2 * z2 ** 2 + y1 ** 2 * z2 ** 2 - 2 * y1 ** 2 * z2 * z3 + y1 ** 2 * z3 ** 2 - 2 * y1 * y2 * z1 * z2 + 2 * y1 * y2 * z1 * z3 + 2 * y1 * y2 * z2 * z3 - 2 * y1 * y2 * z3 ** 2 + 2 * y1 * y3 * z1 * z2 - 2 * y1 * y3 * z1 * z3 - 2 * y1 * y3 * z2 ** 2 + 2 * y1 * y3 * z2 * z3 + y2 ** 2 * z1 ** 2 - 2 * y2 ** 2 * z1 * z3 + y2 ** 2 * z3 ** 2 - 2 * y2 * y3 * z1 ** 2 + 2 * y2 * y3 * z1 * z2 + 2 * y2 * y3 * z1 * z3 - 2 * y2 * y3 * z2 * z3 + y3 ** 2 * z1 ** 2 - 2 * y3 ** 2 * z1 * z2 + y3 ** 2 * z2 ** 2))

    return alpha, beta, gamma, delta


@lru_cache(maxsize=CACHE_SIZE)
def distance_between_edge_and_point(edge, point):
    point1, point2 = tuple(edge.get_points())
    v_vector = np.asarray(point2.get_coordinates()) - np.asarray(point1.get_coordinates())
    w_vector = np.asarray(point.get_coordinates()) - np.asarray(point1.get_coordinates())

    c1 = np.dot(w_vector, v_vector)
    if c1 <= 0:
        return distance_between_points(point, point1)

    c2 = np.dot(v_vector, v_vector)
    if c2 <= c1:
        return distance_between_points(point, point2)

    b = c1 / c2
    nearest_point_coords = np.asarray(point1.get_coordinates()) + b * v_vector
    nearest_point = Point(nearest_point_coords[0], nearest_point_coords[1], nearest_point_coords[2])
    return distance_between_points(point, nearest_point)


@lru_cache(maxsize=CACHE_SIZE)
def distance_between_edges(edge1, edge2):
    vector_u = np.asarray(edge1.get_points()[1].get_coordinates()) - np.asarray(edge1.get_points()[0].get_coordinates())
    vector_v = np.asarray(edge2.get_points()[1].get_coordinates()) - np.asarray(edge2.get_points()[0].get_coordinates())
    vector_w = np.asarray(edge1.get_points()[0].get_coordinates()) - np.asarray(edge2.get_points()[0].get_coordinates())
    a = np.dot(vector_u, vector_u)
    b = np.dot(vector_u, vector_v)
    c = np.dot(vector_v, vector_v)
    d = np.dot(vector_u, vector_w)
    e = np.dot(vector_v, vector_w)
    D = a * c - b * b
    sc, sN, sD = D, D, D
    tc, tN, tD = D, D, D

    if D < PARALLEL_TOLERANCE:
        sN = 0
        sD = 1
        tN = e
        tD = c
    else:
        sN = b * e - c * d
        tN = a * e - b * d
        if sN < 0:
            sN = 0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c

    if tN < 0:
        tN = 0
        if -d < 0:
            sN = 0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if -d + b < 0:
            sN = 0
        elif -d + b > a:
            sN = sD
        else:
            sN = -d + b
            sD = a

    sc = 0 if math.fabs(sN) < PARALLEL_TOLERANCE else sN / sD
    tc = 0 if math.fabs(tN) < PARALLEL_TOLERANCE else tN / tD

    vector_dp = vector_w + (sc * vector_u) - (tc * vector_v)
    distance = math.sqrt(np.dot(vector_dp, vector_dp))
    return distance


@lru_cache(maxsize=CACHE_SIZE)
def distance_between_face_and_edge(face, edge):
    distances = []

    edge_points = edge.get_points()
    for edge_point in edge_points:
        distance = distance_between_face_and_point(face, edge_point)
        distances.append(distance)

    face_edges = face.get_edges()
    for face_edge in face_edges:
        distance = distance_between_edges(face_edge, edge)
        distances.append(distance)

    min_distance = min(distances)
    return min_distance


@lru_cache(maxsize=CACHE_SIZE)
def distance_between_faces(face1, face2):
    distances = []

    for face_edge in face1.get_edges():
        distances.append(distance_between_face_and_edge(face2, face_edge))

    for face_edge in face2.get_edges():
        distances.append(distance_between_face_and_edge(face1, face_edge))

    min_distance = min(distances)
    return min_distance


class Solid:
    def __init__(self, faces):
        self._faces = faces
        self._hashcode = hash(tuple(faces))

    def get_faces(self):
        return self._faces

    def __hash__(self):
        return self._hashcode


class BoundingBox:
    def __init__(self, minx, maxx, miny, maxy, minz, maxz):
        self._min_x = minx
        self._max_x = maxx
        self._min_y = miny
        self._max_y = maxy
        self._min_z = minz
        self._max_z = maxz


def distance_between_face_and_solid(face, solid):
    return min(map(lambda face1: distance_between_faces(face, face1), solid.get_faces()))


def distance_between_solids(solid1, solid2):
    cpus = cpu_count() + 1

    with Pool(processes=cpus) as worker:
        tasks = []
        for face1 in solid1.get_faces():
            task1 = worker.apply_async(distance_between_face_and_solid, (face1, solid2))
            tasks.append(task1)

        return min(map(lambda task: task.get(), tasks))


def create_bounding_box(solid):
    min_x = min(map(lambda face: min(map(lambda point: point.x(), face.get_points())), solid.get_faces()))
    min_y = min(map(lambda face: min(map(lambda point: point.y(), face.get_points())), solid.get_faces()))
    min_z = min(map(lambda face: min(map(lambda point: point.z(), face.get_points())), solid.get_faces()))

    max_x = max(map(lambda face: max(map(lambda point: point.x(), face.get_points())), solid.get_faces()))
    max_y = max(map(lambda face: max(map(lambda point: point.y(), face.get_points())), solid.get_faces()))
    max_z = max(map(lambda face: max(map(lambda point: point.z(), face.get_points())), solid.get_faces()))

    return BoundingBox(min_x, max_x, min_y, max_y, min_z, max_z)


def load_solid(file_name):
    points = []

    with open(file_name, "r", encoding="utf-8") as solid_file:
        for line in solid_file:
            if len(line.strip()) == 0:
                continue
            else:
                x, y, z = tuple(map(lambda num: float(num.strip()), line.split(";")))
                points.append(Point(x, y, z))
    if len(points) % 3 != 0:
        raise Exception
    else:
        def chunks(l, n):
            for i in range(0, len(l), n):
                yield tuple(l[i:i + n])

        points_per_face = list(chunks(points, 3))
        faces = []
        for points in points_per_face:
            p1, p2, p3 = points
            face = Face(p1, p2, p3)
            faces.append(face)
        return Solid(faces)


if __name__ == "__main__":
    solid1 = load_solid("solid1.txt")
    solid2 = load_solid("solid2.txt")
    distance = distance_between_solids(solid1, solid2)
    print("distance between solids = ", distance)

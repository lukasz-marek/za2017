import math
from sympy import Symbol
from functools import lru_cache
from sympy.solvers import solve
import numpy as np
from scipy.optimize import minimize

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

    def __hash__(self):
        return self._hashcode


@lru_cache(maxsize=CACHE_SIZE)
def distance_between_face_and_point(face, point):
    def point_belongs_to_face(point):
        x = Symbol('x')
        y = Symbol('y')
        a, b, c = tuple(face.get_points())
        x_equation = x * (a.x() - c.x()) + y * (b.x() - c.x()) - point.x() + c.x()
        y_equation = x * (a.y() - c.y()) + y * (b.y() - c.y()) - point.y() + c.y()
        z_equation = x * (a.z() - c.z()) + y * (b.z() - c.z()) - point.z() + c.z()
        systems = [[x_equation, y_equation], [x_equation, z_equation], [y_equation, z_equation]]
        analyzed_data = []
        for system in systems:
            analyzed_data = solve(system, x, y)
            if len(analyzed_data) > 0:
                break
        x, y = analyzed_data[x], analyzed_data[y]
        return x >= 0 and y >= 0 and x + y <= 1

    alpha, beta, gamma, delta = compute_plane_equation(face)
    x, y, z = point.get_coordinates()
    c = Symbol('c')
    point_equation = alpha * (x + alpha * c) + beta * (y + beta * c) + gamma * (z + gamma * c) + delta
    c = solve(point_equation, c)[0]
    nearest_point = Point(x + alpha * c, y + beta * c, z + gamma * c)
    if point_belongs_to_face(nearest_point):
        return distance_between_points(nearest_point, point)
    else:
        vertex_a, vertex_b, vertex_c = face.get_points();
        edges = [Edge(vertex_a, vertex_b), Edge(vertex_a, vertex_c), Edge(vertex_b, vertex_c)]
        minimal_distance = math.inf
        for edge in edges:
            distance = distance_between_edge_and_point(edge, point)
            if distance < minimal_distance:
                minimal_distance = distance
        return minimal_distance


@lru_cache(maxsize=CACHE_SIZE)
def compute_plane_equation(face):
    alpha = Symbol('alpha')
    beta = Symbol('beta')
    gamma = Symbol('gamma')
    delta = Symbol('delta')

    system = []
    for face_point in face.get_points():
        x, y, z = face_point.get_coordinates()
        equation = x * alpha + y * beta + z * gamma + delta
        system.append(equation)

    constraint_equation = alpha ** 2 + beta ** 2 + gamma ** 2 - 1
    system.append(constraint_equation)
    plane = solve(system, alpha, beta, gamma, delta)[0]
    return tuple(map(lambda number: number.evalf(), plane))


@lru_cache(maxsize=CACHE_SIZE)
def distance_between_faces(face1, face2):
    alpha1, beta1, gamma1, delta1 = compute_plane_equation(face1)
    alpha2, beta2, gamma2, delta2 = compute_plane_equation(face2)
    start1 = face1.get_points()[0]
    start2 = face2.get_points()[0]
    x1, y1, z1 = start1.get_coordinates()
    x2, y2, z2 = start2.get_coordinates()
    initial_guess = np.asarray([x1, y1, z1, x2, y2, z2])
    optimized_function = lambda g: (g[0] - g[3]) ** 2 + (g[1] - g[4]) ** 2 + (g[2] - g[5]) ** 2

    min_x1 = min(map(lambda point: point.get_coordinates()[0], face1.get_points()))
    min_x2 = min(map(lambda point: point.get_coordinates()[0], face2.get_points()))

    max_x1 = max(map(lambda point: point.get_coordinates()[0], face1.get_points()))
    max_x2 = max(map(lambda point: point.get_coordinates()[0], face2.get_points()))

    min_y1 = min(map(lambda point: point.get_coordinates()[1], face1.get_points()))
    min_y2 = min(map(lambda point: point.get_coordinates()[1], face2.get_points()))

    max_y1 = max(map(lambda point: point.get_coordinates()[1], face1.get_points()))
    max_y2 = max(map(lambda point: point.get_coordinates()[1], face2.get_points()))

    min_z1 = min(map(lambda point: point.get_coordinates()[2], face1.get_points()))
    min_z2 = min(map(lambda point: point.get_coordinates()[2], face2.get_points()))

    max_z1 = max(map(lambda point: point.get_coordinates()[2], face1.get_points()))
    max_z2 = max(map(lambda point: point.get_coordinates()[2], face2.get_points()))

    bounds = ((min_x1, max_x1), (min_y1, max_y1), (min_z1, max_z1),
              (min_x2, max_x2), (min_y2, max_y2), (min_z2, max_z2))

    plane1_constraint = lambda g: g[0] * alpha1 + g[1] * beta1 + g[2] + gamma1 + delta1
    plane2_constraint = lambda g: g[3] * alpha2 + g[4] * beta2 + g[5] + gamma2 + delta2
    constraints = {'type': 'eq', 'fun': plane1_constraint}, {'type': 'eq', 'fun': plane2_constraint}
    result = minimize(optimized_function, initial_guess, method="SLSQP", bounds=bounds, constraints=constraints,
                      options={'maxiter': 5000, 'eps': 1e-20, 'ftol': 1e-20})
    return math.sqrt(result.fun)


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
    D = a*c - b*b
    sc, sN, sD = D, D, D
    tc, tN, tD = D, D, D

    if D < PARALLEL_TOLERANCE:
        sN = 0
        sD = 1
        tN = e
        tD = c
    else:
        sN = b*e - c*d
        tN = a*e - b*d
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

    sc = 0 if math.fabs(sN) < PARALLEL_TOLERANCE else sN/sD
    tc = 0 if math.fabs(tN) < PARALLEL_TOLERANCE else tN/tD

    vector_dp = vector_w + (sc * vector_u) - (tc * vector_v)
    distance = np.linalg.norm(vector_dp)
    return distance


if __name__ == "__main__":
    dist = math.pi
    point_a = Point(10, 0, 0 + dist)
    point_b = Point(0, 10, 0 + dist)
    point_c = Point(0, 0, 10 + dist)
    point_p = Point(3000, 11000, 300)
    face1 = Face(point_a, point_b, point_c)
    point_a2 = Point(10, 0, 0)
    point_b2 = Point(0, 10, 0)
    point_c2 = Point(0, 0, 10)
    face2 = Face(point_a2, point_b2, point_c2)

    # print(distance_between_points(point_a, point_b))
    # print(distance_between_face_and_point(face1, point_p))
    # print(distance_between_faces(face1, face2))
    # print(distance_between_faces_sym(face1, face2))
    edge1 = Edge(point_a, point_p)
    edge2 = Edge(point_b, point_c)
    print(distance_between_edges(edge1, edge2))
    print(distance_between_edges(edge2, edge1))


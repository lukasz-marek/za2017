import math
from sympy import var, sqrt
from sympy.solvers import solve


class Point:
    def __init__(self, x, y, z):
        self._x = x
        self._y = y
        self._z = z

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


def distance_between_points(point1, point2):
    coordinates1 = point1.get_coordinates()
    coordinates2 = point2.get_coordinates()
    squared_distance = (coordinates1[0] - coordinates2[0]) ** 2 \
                       + (coordinates1[1] - coordinates2[1]) ** 2 \
                       + (coordinates1[2] - coordinates2[2]) ** 2
    distance = math.sqrt(squared_distance)
    return distance


class Edge:
    def __init__(self, point1, point2):
        self._point1 = point1
        self._point2 = point2

    def get_points(self):
        return [self._point1, self._point2]


class Face:
    def __init__(self, point1, point2, point3):
        self._point1 = point1
        self._point2 = point2
        self._point3 = point3

    def get_points(self):
        return [self._point1, self._point2, self._point3]


def distance_between_face_and_point(face, point):
    def compute_plane_equation():
        alpha = var('alpha')
        beta = var('beta')
        gamma = var('gamma')
        delta = var('delta')

        system = []
        for face_point in face.get_points():
            x, y, z = face_point.get_coordinates()
            equation = x * alpha + y * beta + z * gamma + delta
            system.append(equation)

        constraint_equation = alpha ** 2 + beta ** 2 + gamma ** 2 - 1
        system.append(constraint_equation)
        plane = solve(system, alpha, beta, gamma, delta)[0]
        return tuple(map(lambda number: number.evalf(), plane))

    def point_belongs_to_face(point):
        x = var('x')
        y = var('y')
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

    alpha, beta, gamma, delta = compute_plane_equation()
    x, y, z = point.get_coordinates()
    c = var('c')
    point_equation = alpha * (x + alpha * c) + beta * (y + beta * c) + gamma * (z + gamma * c) + delta
    c = solve(point_equation, c)[0]
    nearest_point = Point(x + alpha * c, y + beta * c, z + gamma * c)
    if point_belongs_to_face(nearest_point):
        return distance_between_points(nearest_point, point)
    else:
        a, b, c = tuple(face.get_points())
        edges = [Edge(a,b), Edge(a,c), Edge(b, c)]
        distance = min(map(lambda e: distance_between_edge_and_point(e, point), edges))
        return distance


def distance_between_edge_and_point(edge, point):
    point1, point2 = tuple(edge.get_points())
    distance_between_extreme_points = distance_between_points(point1, point2)
    distance = 0
    while distance_between_extreme_points > 1e-6:
        distance_from_1 = distance_between_points(point1, point)
        distance_from_2 = distance_between_points(point2, point)
        middle = Point((point1.x() + point2.x())/2, (point1.y() + point2.y())/2, (point1.z() + point2.z())/2)
        if distance_from_1 < distance_from_2:
            point2 = middle
            distance = distance_from_1
        else:
            point1 = middle
            distance = distance_from_2
        distance_between_extreme_points = distance_between_points(point1, point2)

    return distance


if __name__ == "__main__":
    point_a = Point(0, 0, 0)
    point_b = Point(10000, 0, 0)
    point_c = Point(0, 10000, 0)
    point_p = Point(70, 7, 33)
    face = Face(point_a, point_b, point_c)
    print(distance_between_points(point_a, point_b))
    print(distance_between_face_and_point(face, point_p))
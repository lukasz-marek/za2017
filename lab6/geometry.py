import math
from sympy import var, sqrt
from sympy.solvers.polysys import solve_poly_system


class Point:
    def __init__(self, x, y, z):
        self._x = x
        self._y = y
        self._z = z

    def get_coordinates(self):
        return self._x, self._y, self._z


def distanceBetweenPoints(point1, point2):
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


def distanceBetweenFaceAndPoint(face, point):
    # compute equation of the plane that face belongs to
    alpha = var('alpha')
    beta = var('beta')
    gamma = var('gamma')
    delta = var('delta')

    system = []
    for face_point in face.get_points():
        x, y, z = face_point.get_coordinates()
        equation = x * alpha + y * beta + z * gamma + delta
        system.append(equation)

    constraint_equation = alpha ** 2 + beta ** 2 + delta ** 2 - 1
    system.append(constraint_equation)
    plane = solve_poly_system(system, alpha, beta, gamma, delta)[0]
    alpha, beta, gamma, delta = tuple(map(lambda number: number.evalf(),plane))
    #compute coordinates of nearest point on plane
    x, y, z = point.get_coordinates()
    distance_from_plane = math.fabs(alpha * x + beta * y + gamma * z + delta)

    system.clear()
    nx = var('nx')
    ny = var('ny')
    nz = var('nz')
    a1, a2, a3 = face.get_points()[0].get_coordinates()

    cos_equation = (nx - a1) ** 2 \
                       + (ny - a2) ** 2 \
                       + (nz - a3) ** 2

    distance_equation = distance_from_plane ** 2 - (x - nx) ** 2 - (y - ny) ** 2 - (z - nz) ** 2
    plane_equation = alpha * nx + beta * ny + gamma * nz + delta
    nearest_point = solve_poly_system([distance_equation, plane_equation, cos_equation], nx, ny, nz)
    print(nearest_point)
    return distance_from_plane


if __name__ == "__main__":
    point_a = Point(7, 2, 3)
    point_b = Point(4, 58, 6)
    point_c = Point(7, 8, 9)
    point_p = Point(10, 10, 10)
    face = Face(point_a, point_b, point_c)
    print(distanceBetweenPoints(point_a, point_b))
    print(distanceBetweenFaceAndPoint(face, point_p))
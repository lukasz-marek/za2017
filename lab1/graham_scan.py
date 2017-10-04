import math

EPSILON = 0


def find_new_origin(coordinate_tuples_list):
    # origin - min y, min x (bottom left)
    new_origin = None
    for x, y in coordinate_tuples_list:
        if not new_origin:
            new_origin = (x, y)
        else:
            origin_x, origin_y = new_origin
            if y < origin_y or (y == origin_y and origin_x > x):
                new_origin = (x, y)
    return new_origin


def create_cartesian_to_polar_converter(origin_tuple):
    origin_x, origin_y = origin_tuple

    def convert(coordinate_tuple):
        x, y = coordinate_tuple
        r = math.sqrt((x - origin_x) ** 2 + (y - origin_y) ** 2)
        phi = math.atan((y - origin_y) / (x - origin_x))
        return phi, r

    return convert


def prepare_coordinate_data(origin, coordinate_tuples_list):
    # remove origin
    coordinates = coordinate_tuples_list[::]
    coordinates.remove(origin)
    # convert to polar
    coordinates_conversion_dict = {}
    converter = create_cartesian_to_polar_converter(origin)
    for coordinates_tuple in coordinates:
        coordinates_conversion_dict[coordinates_tuple] = converter(coordinates_tuple)

    # sort coordinates
    sorted_coordinates = sorted(coordinates, key=lambda coordinate: coordinates_conversion_dict[coordinate])

    # if phi's are equal, keep only the one paired with max r
    checked_phi = set()
    results = []
    for coordinates_tuple in sorted_coordinates:
        phi, _ = coordinates_conversion_dict[coordinates_tuple]
        if phi not in checked_phi:
            with_equal_phi = filter(
                lambda coordinate: math.isclose(coordinates_conversion_dict[coordinate][0], phi, rel_tol=EPSILON),
                sorted_coordinates)
            with_equal_phi_and_max_r = max(with_equal_phi,
                                           key=lambda coordinate: coordinates_conversion_dict[coordinate][1])
            checked_phi.add(phi)
            results.append(with_equal_phi_and_max_r)
    return results


def is_left(next_to_top, top, point):
    x1, y1 = next_to_top
    x2, y2 = top
    x, y = point
    d = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    return d < 0


def graham_scan(coordinates_list):
    # if list doesn't containt enough points or is None - return it
    if coordinates_list is None or len(coordinates_list) < 3:
        return coordinates_list

    # remove duplicates
    coordinates = list(set(coordinates_list))
    origin = find_new_origin(coordinates)
    new_coordinates = prepare_coordinate_data(origin, coordinates)

    # algorithm requires 3 or more points but origin is a point as well, so check for three
    if len(new_coordinates) >= 2:
        # functions defined by Cormen
        top = lambda x: x[0]
        next_to_top = lambda x: x[1]

        results = []
        results.insert(0, origin)
        results.insert(0, new_coordinates.pop(0))
        results.insert(0, new_coordinates.pop(0))

        for coordinate in new_coordinates:
            while not is_left(coordinate, next_to_top(results), top(results)):
                results.pop(0)
            results.insert(0, coordinate)

        return results
    else:
        return coordinates


if __name__ == "__main__":
    coords = [(-2, -2), (-5, -3), (-10, -4), (-9, -3), (1, -3), (1, 1), (2, 2), (-4, 1), (-4, 4), (0, 0)]
    print(graham_scan(coords))

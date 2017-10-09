import math
import lab1.parser as parser

EPSILON = 0


def find_new_origin(coordinate_tuples_list):
    # origin - min y, min x (bottom left)
    return min(coordinate_tuples_list)


def create_cartesian_to_polar_converter(origin_tuple):
    origin_x, origin_y = origin_tuple

    def convert(coordinate_tuple):
        x, y = coordinate_tuple
        r = math.sqrt((x - origin_x) ** 2 + (y - origin_y) ** 2)
        phi = math.atan((y - origin_y) / (x - origin_x))
        return phi, r

    return convert


def sort_and_remove_duplicates(coordinate_tuples_list, coordinates_conversion_dict):
    # if phi's are equal, keep only the one paired with max r
    checked_phi = set()
    results = []
    for coordinates_tuple in coordinate_tuples_list:
        phi, _ = coordinates_conversion_dict[coordinates_tuple]
        if phi not in checked_phi:
            with_equal_phi = filter(
                lambda coordinate: math.isclose(coordinates_conversion_dict[coordinate][0], phi, rel_tol=EPSILON),
                coordinate_tuples_list)
            with_equal_phi_and_max_r = max(with_equal_phi,
                                           key=lambda coordinate: coordinates_conversion_dict[coordinate][1])
            checked_phi.add(phi)
            results.append(with_equal_phi_and_max_r)

    # sort coordinates
    sorted_coordinates = list(sorted(results, key=lambda coordinate: coordinates_conversion_dict[coordinate][0]))

    return sorted_coordinates


def prepare_coordinate_data(origin, coordinate_tuples_list):
    # remove origin
    coordinates = coordinate_tuples_list[::]
    coordinates.remove(origin)
    # convert to polar
    coordinates_conversion_dict = {}
    converter = create_cartesian_to_polar_converter(origin)
    for coordinates_tuple in coordinates:
        coordinates_conversion_dict[coordinates_tuple] = converter(coordinates_tuple)

    return sort_and_remove_duplicates(coordinates, coordinates_conversion_dict)


def is_left(point, next_to_top, top):
    x1, y1 = point
    x2, y2 = next_to_top
    x, y = top
    d = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    return d < 0


def graham_scan(coordinates_list):
    # if list doesn't contain enough points or is None - return it
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
    coords = parser.parse()
    result = graham_scan(coords)
    print(result)
    print("Size of results: ", len(result))

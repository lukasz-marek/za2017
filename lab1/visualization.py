from lab1.graham_scan import graham_scan, find_new_origin, prepare_coordinate_data
from lab1.parser import parse
from matplotlib import pyplot as plt


if __name__ == "__main__":
    points = parse()

    origin = find_new_origin(points)
    print("Origin: ", origin)

    preprocessed = prepare_coordinate_data(origin, points)

    result = graham_scan(points)

    for x, y in points:
        plt.plot(x, y, marker='.', ls='', color='black')

    for x, y in preprocessed:
        plt.plot(x, y, marker='X', ls='', color='green')

    segments = result[::]
    segments.append(result[0])

    for i in range(len(segments) - 1):
        start = segments[i]
        end = segments[i + 1]
        plt.plot([start[0], end[0]], [start[1], end[1]], 'r')

    for x, y in result:
        plt.plot(x, y, marker='o', ls='', color='red')

    plt.plot(*origin, "", marker="*", ls="", color="yellow")

    for coordinate in result:
        print(coordinate)

    plt.autoscale()
    plt.show()


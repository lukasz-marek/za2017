from lab1.graham_scan import graham_scan, find_new_origin
from lab1.parser import parse
from matplotlib import pyplot as plt

if __name__ == "__main__":
    points = parse()

    origin = find_new_origin(points)

    plt.plot(origin, marker="*", ls="", color="yellow")

    plt.plot(points, marker='.', ls='', color='black')

    result = graham_scan(points)
    plt.plot(result, marker='o', ls='', color='red')

    for coords in result:
        print(coords)

    plt.autoscale()
    plt.show()


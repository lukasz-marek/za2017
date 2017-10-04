def parse():
    results = []
    with open("punktyPrzykladowe.csv","r",encoding="utf-8") as input_file:
        for line in input_file:
            parts = line.split(",")
            x = float(parts[0])
            y = float(parts[1])
            results.append((x, y))
    return results

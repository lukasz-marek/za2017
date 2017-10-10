import binarytree


def compute_letter_counts(filepath):
    results = {}
    with open(filepath,"r",encoding="utf-8") as file:
        for line in file:
            for character in line:
                if character in results:
                    results[character] += 1
                else:
                    results[character] = 1
    return results
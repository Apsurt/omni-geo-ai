import csv

class Converter:
    def __init__(self):
        self._polydatapoints = []

    def csv_to_pointstring(self, path) -> str:
        with open(path, 'r', encoding="UTF-8") as f:
            read = csv.reader(f)
            for elem in read:
                self._polydatapoints.append(elem[0][6:].replace(' ', ', '))
        # self._polydatapoints.pop(0)
        string_coords = "["
        for elem in self._polydatapoints:
            string_coords += elem + ","
        string_coords += "]"
        return string_coords

def main():
    a = Converter()
    tocopy = a.csv_to_pointstring("./omni/resources/pins.csv")
    print(tocopy)

if __name__ == "__main__":
    main()

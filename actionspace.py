class ActionSpace():
    def __init__(self, y, x) -> None:
        self.__actionspace: dict = dict()
        self.__key_map: dict = dict()
        self.__size: int = 0
        self.calculate(y, x)

    def __add(self, key, obj) -> None:
        self.__actionspace[key] = obj
        self.__key_map[obj] = key

    def calculate(self, y, x) -> None:
        key = 0
        for ys in range(y):
            for xs in range(x):
                for yt in range(y):
                    for xt in range(x):
                        if (ys, xs) == (yt, xt):
                            continue
                        self.__add(key, (ys, xs, yt, xt))
                        key += 1
        self.__size = key

    def __getitem__(self, key) -> tuple:
        return self.__actionspace[key]

    def get_key(self, obj) -> int:
        return self.__key_map[obj]

    @property
    def size(self) -> int:
        return self.__size

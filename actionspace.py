class ActionSpace():
    def __init__(self) -> None:
        self.__actionspace: dict = dict()
        self.__key_map: dict = dict()
        self.__size: int = 0

    def __getitem__(self, key) -> tuple:
        return self.__actionspace[key]
    
    def add(self, value) -> None:
        if value in self.__key_map:
            return
        self.__actionspace[self.__size] = value
        self.__key_map[value] = self.__size
        self.__size += 1

    def get_key(self, obj) -> int:
        return self.__key_map[obj]

    @property
    def size(self) -> int:
        return self.__size

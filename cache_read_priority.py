import sys
from collections import OrderedDict
from contextlib import contextmanager
from threading import Condition, Lock


class Cache:
    def __init__(self, max_size=100) -> None:
        """Thread safe LRU cache, read priority, set max size in MB"""
        self.__read_ready = Condition(Lock())
        self.__writers: int = 0
        self.__readers: int = 0
        self.__size: int = 0
        self.__max_size: int = max_size * 1024 * 1024
        self.__cache = OrderedDict()

    def __acquire_read(self):
        self.__read_ready.acquire()
        try:
            self.__readers += 1
            while self.__writers > 0:
                self.__read_ready.wait()
        finally:
            self.__read_ready.release()

    def __release_read(self):
        self.__read_ready.acquire()
        try:
            self.__readers -= 1
            if self.__readers == 0:
                self.__read_ready.notify_all()
        finally:
            self.__read_ready.release()

    def __acquire_write(self):
        self.__read_ready.acquire()
        try:
            while self.__readers > 0 or self.__writers > 0:
                self.__read_ready.wait()
            self.__writers += 1
        finally:
            self.__read_ready.release()

    def __release_write(self):
        self.__read_ready.acquire()
        try:
            self.__writers -= 1
            self.__read_ready.notify_all()
        finally:
            self.__read_ready.release()

    @contextmanager
    def __read(self):
        self.__acquire_read()
        yield
        self.__release_read()

    @contextmanager
    def __write(self):
        self.__acquire_write()
        yield
        self.__release_write()

    def __contains__(self, key) -> bool:
        with self.__read():
            return key in self.__cache

    def __getitem__(self, key) -> object:
        with self.__read():
            self.__cache.move_to_end(key, last=True)
            return self.__cache[key]

    def __len__(self) -> int:
        with self.__read():
            return len(self.__cache)

    def __setitem__(self, key, value) -> None:
        with self.__write():
            if key in self.__cache:
                if not self.__remove(key):
                    return
            if self.__size >= self.__max_size:
                if not self.__evict():
                    return
            self.__cache[key] = value
            self.__size += sys.getsizeof((key, value))

    def add(self, key, value) -> None:
        """Add a key-value pair to the cache, respecting the max size"""

        with self.__write():
            if self.__size >= self.__max_size:
                if not self.__evict():
                    return
            self.__cache[key] = value
            self.__size += sys.getsizeof((key, value))

    def clear(self) -> None:
        """Clear the cache"""

        with self.__write():
            self.__cache.clear()
            self.__size = 0

    def __evict(self) -> None:
        """Remove the least recently used key-value pair from the cache"""
        with self.__write():
            try:
                tpl = self.__cache.popitem(last=False)
            except KeyError:
                return False
            self.__size -= sys.getsizeof(tpl)
            return True

    def __remove(self, key) -> None:
        """Remove a key-value pair from the cache"""
        with self.__write():
            try:
                value = self.__cache.pop(key)
            except KeyError:
                return False
            self.__size -= sys.getsizeof((key, value))
            return True

import sqlite3
import random
import multiprocessing as mp
import numpy as np


class bit_array():
    def __init__(self, size: int) -> None:
        self.__size = size
        self.__bit_array = np.zeros(
            np.ceil(size/64).astype(np.uint64), dtype=np.uint64)

    def __getitem__(self, index: int) -> bool:
        if index >= self.__size:
            raise IndexError("Index out of range")
        return bool(self.__bit_array[index//64] & np.uint64(1 << (index % 64)))

    def __setitem__(self, index: int, value: bool) -> None:
        if index >= self.__size:
            raise IndexError("Index out of range")
        if value:
            self.__bit_array[index//64] |= np.uint64(1 << (index % 64))
        else:
            self.__bit_array[index//64] &= np.uint64(~(1 << (index % 64)))


class db_dataloader(mp.Process):
    def __init__(self, db_path, table_name, num_batches, batch_size,  min_index,
                 max_index, random=True, replace=True, shuffle=True, slice_size=32) -> None:
        super().__init__()
        self.__db_path = db_path
        self.__table_name = table_name
        self.__batch_size = batch_size
        self.__num_batches = num_batches
        self.__random = random
        self.__replace = replace
        self.__shuffle = shuffle
        self.__min_index = min_index
        self.__max_index = max_index
        self.__slice_size = slice_size
        self.__batch_buffer = mp.Queue(maxsize=2)
        self.batches_left = mp.Value('i', num_batches)

    def __get_slice(self, idx):
        # Open a new connection inside the child process
        conn = sqlite3.connect(self.__db_path)
        cursor = conn.cursor()
        end_idx = idx + self.__slice_size
        db_slice = cursor.execute(
            "SELECT * FROM {} WHERE id >= {} AND id < {}".format(self.__table_name, idx, end_idx)).fetchall()
        conn.close()
        return db_slice

    def __create_batch(self):
        batch = []
        while len(batch) < self.__batch_size:
            batch.extend(self.__get_slice(next(self.__idx_generator)))
        batch = np.array(batch)
        if self.__shuffle:
            np.random.shuffle(batch)
        self.__batch_buffer.put(batch)

    def run(self):
        for i in range(self.__num_batches):
            self.__create_batch()

    @property
    def __random_idx(self):
        return random.randint(self.__min_index, self.__max_index//self.__slice_size)

    @property
    def __next_idx(self):
        for idx in range(self.__min_index, self.__max_index//self.__slice_size):
            yield idx * self.__slice_size
        raise StopIteration

    @property
    def __no_replace_idx(self):
        # create a mask for the random indices
        mask_range = np.ceil((
            self.__max_index - self.__min_index) / self.__slice_size)
        replace_masker = bit_array(mask_range)
        idx_drawn = 0

        # draw random indices until 80% of the indices are drawn
        while self.__max_index * 0.8 > idx_drawn:
            idx = self.__random_idx
            if replace_masker[idx]:
                continue
            replace_masker[idx] = 1
            idx_drawn += 1
            yield idx * self.__slice_size

        # draw the remaining indices
        for idx in range(self.__min_index, self.__max_index//self.__slice_size):
            if replace_masker[idx]:
                continue
            replace_masker[idx] = 1
            yield idx * self.__slice_size

    @property
    def __idx_generator(self):
        if not self.__random:
            yield from self.__next_idx
        elif not self.__replace:
            yield from self.__no_replace_idx
        else:
            yield self.__random_idx * self.__slice_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.batches_left.value <= 0:
            raise StopIteration
        self.batches_left.value -= 1
        return self.__batch_buffer.get(timeout=60)

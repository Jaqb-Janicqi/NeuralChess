import sqlite3
import random
import multiprocessing as mp
import sys
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


class dataloader(mp.Process):
    def __init__(self, db_path, table_name, min_index, max_index, num_batches=0,
                 batch_size=0, slice_size=0, random=True, replace=True, shuffle=True,
                 specials={}) -> None:
        """Dataloader for sqlite databases in a seperate process. Either batch_size or num_batches must be set.
        If database contains blob data and you want to convert it inside dataloader, 
        you must pass a dictionary with the column names as keys and the conversion functions as values.
        """

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
        self.__last_idx = 0
        self.batches_left = mp.Value('i', num_batches)
        self.__specials = specials
        self.__specials_query_pos = {}
        self.__num_query_cols = None
        self.__calculate_batch_and_slice_size()
        self.__get_table_structure()

    def __get_table_structure(self):
        conn = sqlite3.connect(self.__db_path)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({self.__table_name})")
        table_structure = cursor.fetchall()
        conn.close()
        self.__num_query_cols = len(table_structure)
        for row in table_structure:
            if row[1] in self.__specials:
                self.__specials_query_pos[row[1]] = row[0]

    def __calculate_batch_and_slice_size(self):
        if self.__max_index == 0:
            conn = sqlite3.connect(self.__db_path)
            cursor = conn.cursor()
            self.__max_index = cursor.execute(
                "SELECT COUNT(*) FROM {}".format(self.__table_name)).fetchone()[0]
            conn.close()
        if self.__max_index < self.__min_index:
            raise ValueError("max_index must be greater than min_index")
        if self.__batch_size == 0 and self.__num_batches == 0:
            raise ValueError(
                "Either batch_size or num_batches must be greater than 0")
        if self.__batch_size == 0:
            self.__batch_size = np.ceil(
                self.__max_index / self.__num_batches).astype(np.uint64)
        if self.__num_batches == 0:
            self.__num_batches = np.ceil(
                self.__max_index / self.__batch_size).astype(np.uint64)
            self.batches_left.value = self.__num_batches
        if self.__slice_size == 0:
            self.__slice_size = self.__batch_size
        if self.__last_idx < self.__min_index:
            self.__last_idx = self.__min_index

    def __get_slice(self, idx):
        conn = sqlite3.connect(self.__db_path)
        cursor = conn.cursor()
        end_idx = idx + self.__slice_size
        db_slice = cursor.execute(
            "SELECT * FROM {} WHERE id >= {} AND id < {}".format(self.__table_name, idx, end_idx)).fetchall()
        conn.close()
        return db_slice

    def __get_single(self, idx):
        conn = sqlite3.connect(self.__db_path)
        cursor = conn.cursor()
        db_slice = cursor.execute(
            "SELECT * FROM {} WHERE id = {}".format(self.__table_name, idx)).fetchall()
        conn.close()
        return db_slice

    def __create_batch(self):
        batch = []
        missed = 0
        while len(batch) < self.__batch_size:
            idx = self.__generate_idx
            if idx is None:
                return None
            if self.__slice_size == 1:
                slc = self.__get_single(idx)
            else:
                slc = self.__get_slice(idx)
            if len(slc) == 0:
                missed += 1
                if missed > 10:
                    raise StopIteration
                continue
            missed = 0
            if self.__specials != {}:
                for row in slc:
                    new_row = list(row)
                    for key, func in self.__specials.items():
                        new_row[self.__specials_query_pos[key]] = func(
                            new_row[self.__specials_query_pos[key]])
                    slc[slc.index(row)] = new_row
            batch.extend(slc)

        if self.__shuffle:
            random.shuffle(batch)

        # split batch into columns
        out_batch = [[] for _ in range(self.__num_query_cols)]
        for row in batch:
            for i in range(self.__num_query_cols):
                out_batch[i].append(row[i])

        # convert batch to numpy arrays of correct shape
        if self.__batch_size > 1:
            out_batch = [np.asarray(col) for col in out_batch]
        else:
            out_batch = [np.asarray(col)[0] for col in out_batch]
        self.__batch_buffer.put(out_batch)

    def run(self):
        if self.__num_batches == 0:
            self.__num_batches = sys.maxsize ** 10
        for _ in range(self.__num_batches):
            self.__create_batch()

    def __random_idx(self):
        return random.randint(self.__min_index, self.__max_index // self.__slice_size)

    def __next_idx(self):
        if self.__last_idx >= self.__max_index // self.__slice_size:
            return None
        idx = self.__last_idx
        self.__last_idx += 1
        return idx

    def __no_replace_idx(self):
        # create a mask for the random indices
        mask_range = int(
            np.ceil((self.__max_index - self.__min_index) / self.__slice_size))
        replace_masker = bit_array(mask_range)
        idx_drawn = 0

        # draw random indices until 80% of the indices are drawn
        while self.__max_index * 0.8 > idx_drawn:
            idx = self.__random_idx()
            if replace_masker[idx]:
                continue
            replace_masker[idx] = 1
            idx_drawn += 1
            yield idx

        # draw the remaining indices
        for idx in range(self.__min_index, self.__max_index // self.__slice_size):
            if replace_masker[idx]:
                continue
            replace_masker[idx] = 1
            yield idx

    @property
    def __generate_idx(self):
        if not self.__random:
            return self.__next_idx()
        elif not self.__replace:
            return next(self.__no_replace_idx())
        else:
            return self.__random_idx()

    def __iter__(self):
        return self

    def __next__(self):
        if self.batches_left.value <= 0:
            raise StopIteration
        self.batches_left.value -= 1
        try:
            return self.__batch_buffer.get(timeout=5*60)
        except:
            raise StopIteration

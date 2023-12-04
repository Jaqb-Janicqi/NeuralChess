import sqlite3
import random
import multiprocessing as mp
import sys
import numpy as np
import torch
import chess
from actionspace import ActionSpace


ASPACE = ActionSpace()


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


class DataLoader(mp.Process):
    def __init__(self, db_path, table_name, min_index, max_index,
                 num_batches=0, batch_size=0, slice_size=0, buffer_size=8,
                 random=True, replace=True, shuffle=True, to_tensor=True,
                 use_offsets=False,
                 specials={}, output_columns=[]) -> None:
        """DataLoader for sqlite databases in a seperate process. Either batch_size or num_batches must be set.
        If database contains blob data and you want to convert it inside dataloader, 
        you must pass a dictionary with the column names as keys and the conversion functions as values.
        Output columns are returned in the same order they are declared in the list.
        """

        super().__init__(daemon=True)
        self.__db_path = db_path
        self.__table_name = table_name
        self.__batch_size = batch_size
        self.__num_batches = num_batches
        self.__random = random
        self.__replace = replace
        self.__shuffle = shuffle
        self.__to_tensor = to_tensor
        self.__use_offsets = use_offsets
        self.__min_index = int(min_index)
        self.__max_index = int(max_index)
        self.__slice_size = slice_size
        self.__batch_buffer = mp.Queue(maxsize=buffer_size)
        self.__batches_left = 0
        self.__specials = specials
        self.__output_columns = output_columns
        self.__specials_query_pos = {}
        self.__data_col_pos = {}
        self.__num_query_cols = None
        self.__idx_drawn = 0
        self.__last_idx = 0
        self.__idx_drawn = 0
        self.__last_idx_checked = self.__min_index // self.__slice_size
        self.__setup_attrs()
        self.__get_table_structure()
        self.__mask_range = int(
            np.ceil((self.__max_index - self.__min_index) / self.__slice_size))
        self.__restart_latch = mp.Event()
        self.__running = True
        self.__init_completed = False

    def __reinit(self):
        if self.__num_batches == 0:
            self.__num_batches = sys.maxsize ** 10
        self.__idx_drawn = 0
        if not self.__replace:
            self.__replace_masker = bit_array(self.__mask_range)
        self.__last_idx = 0
        self.__idx_drawn = 0
        self.__last_idx_checked = self.__min_index

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
            if row[1] in self.__output_columns:
                self.__data_col_pos[row[1]] = row[0]
        
        ##############################################
        # add legal policy
        if 'policy' in self.__output_columns:
            self.__num_query_cols += 1
            self.__data_col_pos['legal_policy'] = self.__num_query_cols - 1
            self.__output_columns.append('legal_policy')
        ##############################################

    def __setup_attrs(self):
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
        self.__batches_left = self.__num_batches
        if self.__slice_size == 0:
            self.__slice_size = self.__batch_size
        if self.__last_idx < self.__min_index:
            self.__last_idx = self.__min_index

    def __get_slice(self, idx):
        conn = sqlite3.connect(self.__db_path)
        cursor = conn.cursor()
        end_idx = idx + self.__slice_size
        if self.__use_offsets:
            db_slice = cursor.execute(
                "SELECT * FROM {} LIMIT {} OFFSET {}".format(self.__table_name, self.__slice_size, idx)).fetchall()
        else:
            db_slice = cursor.execute(
                "SELECT * FROM {} WHERE id >= {} AND id < {}".format(self.__table_name, idx, end_idx)).fetchall()
        conn.close()
        return db_slice

    def __get_single(self, idx):
        conn = sqlite3.connect(self.__db_path)
        cursor = conn.cursor()
        if self.__use_offsets:
            db_slice = cursor.execute(
                "SELECT * FROM {} LIMIT 1 OFFSET {}".format(self.__table_name, idx)).fetchall()
        else:
            db_slice = cursor.execute(
                "SELECT * FROM {} WHERE id = {}".format(self.__table_name, idx)).fetchall()
        conn.close()
        return db_slice

    def __create_batch(self):
        batch = []
        missed = 0
        while len(batch) < self.__batch_size and missed < 10:
            idx = self.__generate_idx
            if idx is None:
                break
            if self.__slice_size == 1:
                slc = self.__get_single(idx)
            else:
                slc = self.__get_slice(idx)
            if not slc:
                missed += 1
            missed = 0

            # convert slice to list of lists
            slc = [list(row) for row in slc]

            ##############################################
            # create legal policy
            if 'policy' in self.__output_columns:
                for row in slc:
                    row.append(self.get_legal_policy(row[self.__data_col_pos['fen']]))
            ##############################################

            if self.__specials:
                for key, func in self.__specials.items():
                    special_idx = self.__specials_query_pos[key]
                    for row in slc:
                        row[special_idx] = func(row[special_idx])
            batch.extend(slc)

        if not batch:
            return False

        if self.__shuffle:
            random.shuffle(batch)

        # Split batch into columns
        tmp = [[] for _ in range(self.__num_query_cols)]
        for row in batch:
            for i, col in enumerate(row):
                tmp[i].append(col)        

        batch = []
        if self.__to_tensor:
            for key in self.__output_columns:
                column = tmp[self.__data_col_pos[key]]
                column = torch.from_numpy(np.array(column)).squeeze().float()
                batch.append(column)
        else:
            for key in self.__output_columns:
                column = tmp[self.__data_col_pos[key]]
                batch.append(column)

        self.__batch_buffer.put(batch)
        return True

    def run(self):
        while True:
            self.__reinit()
            self.__running = True
            self.__init_completed = True
            for _ in range(self.__num_batches):
                created = self.__create_batch()
                if not created:
                    break
            self.__running = False
            self.__restart_latch.wait()
            self.__restart_latch.clear()

    def __random_idx(self):
        return random.randint(self.__min_index, self.__max_index) // self.__slice_size

    def __next_idx(self):
        if self.__last_idx >= self.__max_index // self.__slice_size:
            return None
        idx = self.__last_idx
        self.__last_idx += 1
        return idx

    def __no_replace_idx(self):
        # draw random indices until 80% of the indices are drawn
        while (self.__max_index - self.__min_index) // self.__slice_size * 0.8 > self.__idx_drawn:
            idx = self.__random_idx() - self.__min_index // self.__slice_size
            if self.__replace_masker[idx]:
                continue
            self.__replace_masker[idx] = 1
            self.__idx_drawn += 1
            return idx

        # draw the remaining indices
        for idx in range(self.__last_idx_checked, int(self.__max_index // self.__slice_size)):
            if self.__replace_masker[idx]:
                continue
            self.__replace_masker[idx] = 1
            self.__idx_drawn += 1
            self.__last_idx_checked = idx
            return idx
        return None

    def restart(self):
        self.__restart_latch.set()

    @property
    def __generate_idx(self):
        if not self.__random:
            return self.__next_idx()
        elif not self.__replace:
            return self.__no_replace_idx()
        else:
            return self.__random_idx()

    def __iter__(self):
        return self

    def __next__(self):
        if self.__batches_left <= 0:
            self.__batches_left = self.__num_batches
            raise StopIteration
        if self.__batch_buffer.empty() and not self.__running and self.__init_completed:
            self.restart()
        self.__batches_left -= 1
        try:
            return self.__batch_buffer.get(timeout=60)
        except:
            raise StopIteration

    def __len__(self):
        if self.__num_batches != 0:
            return self.__num_batches
        return int(np.ceil((self.__min_index - self.__max_index) / self.__batch_size))

    @property
    def dataset(self):
        return self

    def get_legal_policy(self, fen):
        board = chess.Board(fen)
        legal_moves = board.legal_moves
        legal_policy = np.zeros(ASPACE.size, dtype=np.float32)
        for move in legal_moves:
            legal_policy[ASPACE.get_key(move)] = 1
        return legal_policy


def convert_to_numpy(arr):
    return np.frombuffer(arr, dtype=np.float32).reshape(16, 8, 8)


if __name__ == "__main__":
    conn = sqlite3.connect('C:/sqlite_chess_db/lichess.db')
    db_size = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
    conn.close()
    db_checked = 22547088
    x = (db_checked - 1) // 64

    tmp = DataLoader(
        db_path='C:/sqlite_chess_db/lichess.db',
        table_name='positions',
        num_batches=2000,
        batch_size=1024,
        min_index=1,
        max_index=db_size*0.8,
        random=True,
        replace=False,
        shuffle=True,
        slice_size=64,
        specials={'encoded': convert_to_numpy},
        data_cols=['encoded'],
        label_cols=['prob']
    )
    tmp.start()
    corrupt = 0
    bnum = 0
    for batch in tmp:
        print(bnum)
        x, y = batch
        if x.shape[0] == 0 or y.shape[0] == 0:
            corrupt += 1
        bnum += 1
    print()
    print(corrupt)

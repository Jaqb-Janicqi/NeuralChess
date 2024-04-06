import sqlite3
import math
import chess
import chess.pgn
import numpy as np
import os
import tqdm
import time
import io
import torch

import multiprocessing as mp
from actionspace.actionspace import ActionSpace as asp
from helper.helper_functions import *
from collections import OrderedDict
import psutil


UINT32_MAX = 2**32 - 1
MAX_MEM_USAGE = 10*1024*1024*1024   # xGB
DUMP_SIZE = int(1e2)
DUMP_QUEUE_SIZE = int(1e6)
COMMIT_SIZE = int(1e4)
FILE_START = 0
FILE_END = None


class Position:
    def __init__(self, encoded, fen, cp, policy, value, sample_count, legal_moves):
        self.encoded: str = encoded
        self.fen: str = fen
        self.cp: int = cp
        self.policy: np.ndarray = policy
        self.value: float = value
        self.sample_count: int = sample_count
        self.legal_moves: int = legal_moves

    def merge(self, other: 'Position'):
        if self.sample_count + other.sample_count > UINT32_MAX:
            return False
        self.cp = other.cp if abs(other.cp) < abs(self.cp) else self.cp
        self.value = other.value if abs(
            other.value) < abs(self.value) else self.value
        if not self.policy.flags.writeable:
            self.policy = self.policy.copy()
        self.policy += other.policy
        self.sample_count += other.sample_count
        return True

    def update(self, cp, value, move_id):
        if self.sample_count + 1 > UINT32_MAX:
            return False
        self.cp = cp if abs(cp) < abs(self.cp) else self.cp
        self.value = value if abs(value) < abs(self.value) else self.value
        self.policy[move_id] += 1
        self.sample_count += 1
        return True


def get_centipawns(prob):
    return int(111.714640912 * math.tan(1.5620688421 * prob))


def map_centipawns_to_probability(centipawns):
    return math.atan2(centipawns, 111.714640912) / 1.5620688421


def get_position_count(conn):
    return conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]


def get_position(conn, encoded):
    return conn.execute("SELECT * FROM positions WHERE encoded = ?", (encoded,)).fetchone()


def insert_or_replace_position(conn, item: Position):
    # Insert or replace the position
    conn.execute("""
        INSERT OR REPLACE INTO positions (encoded, fen, cp, policy, value, sample_count, legal_moves)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT DO UPDATE SET
            cp = excluded.cp,
            policy = excluded.policy,
            value = excluded.value,
            sample_count = excluded.sample_count
    """, (item.encoded, item.fen, item.cp, item.policy, item.value, item.sample_count, item.legal_moves))


def fix_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    row_id = 1
    while True:
        cursor.execute("SELECT * FROM positions WHERE id = ?", (row_id,))
        row = cursor.fetchone()
        if not row:
            break
        row_id += 1
        encoded = row[1]
        fen = row[2]
        cp = row[3]
        policy = decode_a85(row[4])
        value = row[5]
        sample_count = np.sum(policy)
        legal_moves = row[7]
        item = Position(encoded, fen, cp, policy, value,
                        sample_count, legal_moves)
        insert_or_replace_position(conn, item)
    conn.close()


def process_game(line, actionspace: asp, db_dict, verbose=True, rotate=True, white_pov=False):
    try:
        pgn = io.StringIO(line)
        game = chess.pgn.read_game(pgn)
        if not game:
            return
        if game.next() is None:
            return
        # game = game.next()  # skip starting position
        while True:
            move_made = game.next().move
            if rotate and game.board().turn == chess.BLACK:
                # flip board such that the king is always on the right
                fen = game.board().mirror().fen()
                move_made = chess.Move(
                    chess.square_mirror(move_made.from_square),
                    chess.square_mirror(move_made.to_square),
                    move_made.promotion
                )
            else:
                fen = game.board().fen()

            cp = game.eval()
            if cp is None:
                cp = 0
            else:
                if white_pov:
                    cp = cp.white().score(mate_score=12800)
                else:
                    cp = cp.relative.score(mate_score=12800)
            value = map_centipawns_to_probability(cp)
            encoded = encode_a85(decode_from_fen(fen))

            if encoded in db_dict:
                db_dict[encoded].update(
                    cp, value, actionspace.get_key(move_made))
            else:
                policy = np.zeros(actionspace.size, dtype=np.uint32)
                policy[actionspace.get_key(move_made)] += 1
                pos = Position(encoded, fen, cp, policy, value, 1,
                               game.board().legal_moves.count())
                db_dict[encoded] = pos

            # move to next position
            game = game.next()
            if not game:
                return
            if not game.eval():
                return
            if not game.next():
                return
    except Exception as e:
        if verbose:
            print(e)
            print(line)


def parse_worker(shard_path, queue: mp.Queue, rotate, absolute_score):
    db_dict = OrderedDict()
    actionspace = asp()
    skip_game = False
    with open(shard_path) as file:
        for line in file:
            if line == '\n':
                continue

            # check game skip unless skip_game is True
            if skip_game == False:
                if line.startswith('[WhiteElo'):
                    elo = int(line.split('"')[1])
                    if elo < 2200:
                        skip_game = True
                        continue

                if line.startswith('[BlackElo'):
                    elo = int(line.split('"')[1])
                    if elo < 2200:
                        skip_game = True
                        continue

                if line.startswith('[TimeControl'):
                    time_control = line.split('"')[1]
                    total_time = int(time_control.split('+')[0])
                    if total_time < 180:
                        skip_game = True
                        continue

            if line.startswith("1."):
                if skip_game:
                    skip_game = False
                    continue
                if '[%eval' in line:
                    process_game(line, actionspace, db_dict, rotate=rotate, white_pov=absolute_score)

    # dump items
    for _ in range(len(db_dict)):
        _, item = db_dict.popitem(last=False)
        queue.put(item)


def prune_database(conn):
    # remove all rows with sample_count < 2
    conn.execute("DELETE FROM positions WHERE sample_count < 2")


def should_prune(db_path):
    # prune if db is larger than 1TB
    if os.path.getsize(db_path) > 1e12:
        return True
    return False


def write_to_db(conn, item: Position):
    pos = get_position(conn, item.encoded)
    if pos:
        pos = Position(*(pos[1:]))
        pos.policy = decode_a85(pos.policy)
        if not item.merge(pos):
            return
    item.policy = encode_a85(item.policy)
    insert_or_replace_position(conn, item)
    del item, pos


def db_worker(queue: mp.Queue, db_path: str):
    conn = sqlite3.connect(db_path)
    pending_commit = 0
    while True:
        item = queue.get()
        if item is None:
            break

        write_to_db(conn, item)
        pending_commit += 1

        if pending_commit >= COMMIT_SIZE:
            conn.commit()
            pending_commit = 0
    conn.commit()
    conn.close()


def add_item(dictionary, item: Position):
    if item.encoded in dictionary:
        dictionary[item.encoded].merge(item)
        del item
    else:
        dictionary[item.encoded] = item


def data_worker(parser_queue: mp.Queue, db_queue: mp.Queue):
    db_dict = OrderedDict()
    p_id = mp.current_process().pid

    while True:
        mem_usage = psutil.Process(p_id).memory_info().vms
        # start writing to db if mem usage is too high
        if mem_usage > MAX_MEM_USAGE:
            for _ in range(DUMP_SIZE):
                try:
                    _, item = db_dict.popitem(last=False)
                except KeyError:
                    break
                db_queue.put(item)
            continue

        # wait for item from parser
        while parser_queue.empty():
            time.sleep(1)

        # get item from parser
        q_item = parser_queue.get()
        # exit if parsing has finished
        if q_item is None:
            break

        # add item to db_dict
        add_item(db_dict, q_item)

    # dump remaining items
    for _ in range(len(db_dict)):
        _, item = db_dict.popitem(last=False)
        db_queue.put(item)
    db_queue.put(None)


def parse(shard_path) -> None:
    rotate_board = False
    absolute_score = True
    tic = time.time()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            encoded BLOB NOT NULL UNIQUE,
            fen TEXT NOT NULL,
            cp INTEGER NOT NULL,
            policy BLOB NOT NULL,
            value REAL NOT NULL,
            sample_count INTEGER NOT NULL,
            legal_moves INTEGER NOT NULL
        );
    """)
    conn.commit()
    conn.close()
    files = os.listdir(shard_path)
    if FILE_END is not None:
        file_end = FILE_END
    else:
        file_end = len(files)
    files = files[FILE_START:file_end]
    parser_queue = mp.Manager().Queue(int(1e3))
    db_queue = mp.Manager().Queue(DUMP_QUEUE_SIZE)
    db_writer = mp.Process(target=db_worker, args=(db_queue, db_path))
    db_writer.start()
    data_agregator = mp.Process(target=data_worker, args=(
        parser_queue, db_queue))
    data_agregator.start()
    p_size = mp.cpu_count()
    # p_size = 1
    with mp.Pool(processes=p_size) as pool:
        pbar = tqdm.tqdm(total=len(files), desc="Parsing files",
                         leave=False, dynamic_ncols=True, position=0)
        for file in files:
            if file.endswith(".pgn"):
                pool.apply_async(parse_worker, args=(
                    shard_path + file, parser_queue, rotate_board, absolute_score), callback=lambda _: pbar.update())
        pool.close()
        pool.join()
    pbar.close()
    parser_queue.put(None)
    data_agregator.join()
    db_queue.put(None)
    db_writer.join()
    toc = time.time()
    print(f"Processed {len(files)} files in {(toc - tic)/60} minutes")


def convert_to_numpy(arr):
    return np.frombuffer(arr, dtype=np.float32).reshape(16, 8, 8)


def display():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM positions where sample_count > legal_moves order by RANDOM() LIMIT 1000")
    rows = cursor.fetchall()
    print(len(rows))
    for row in rows:
        fen = row[1]
        print(fen)
        print(row[2])
        policy = decode_a85(row[3])
        played = np.argwhere(policy > 0)
        actionspace = asp()
        moves = []
        for move in played:
            moves.append(actionspace[move[0]])
        import matplotlib.pyplot as plt
        import seaborn as sns
        # plot frequency of moves
        sns.set_theme(style="whitegrid")
        sns.set(rc={'figure.figsize': (10, 5)})
        ax = sns.barplot(x=moves, y=policy[played].flatten())
        plt.show(block=True)
    conn.close()


def count():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM positions")
    rows = cursor.fetchall()
    print(rows)
    cursor.execute(
        "SELECT COUNT(*) FROM positions where sample_count > legal_moves or sample_count > 20")
    rows = cursor.fetchall()
    print(rows)
    cursor.execute(
        "SELECT COUNT(*) FROM positions where sample_count > legal_moves or sample_count > 10")
    rows = cursor.fetchall()
    print(rows)
    cursor.execute(
        "SELECT COUNT(*) FROM positions where sample_count > legal_moves or sample_count > 5")
    rows = cursor.fetchall()
    print(rows)
    cursor.execute(
        "SELECT COUNT(*) FROM positions where sample_count > legal_moves or sample_count > 2")
    rows = cursor.fetchall()
    print(rows)
    conn.close()


db_path = 'C:/sqlite_chess_db/lichess2200_whitepov.db'

if __name__ == '__main__':
    # fix_db(db_path)
    # count()
    # display()
    parse('E:/lichess_shards/lichess_db_standard_rated_2023-9/')
    # parse('E:/lichess_shards/lichess_db_standard_rated_2023-10/')

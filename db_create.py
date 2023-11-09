import sqlite3
import math
import chess
import chess.pgn
import numpy as np
import os
import tqdm
import time
import io

import yaml
from mcts_node import Node
import stockfish
# from mcts_node import Node
from mcts_node_ext import Node
import multiprocessing as mp
from actionspace import ActionSpace as asp
import sys
import base64


def get_centipawns(prob):
    return int(111.714640912 * math.tan(1.5620688421 * prob))


def map_centipawns_to_probability(centipawns):
    return math.atan2(centipawns, 111.714640912) / 1.5620688421


def get_position_count(conn):
    return conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]


def get_position(conn, encoded):
    return conn.execute("SELECT * FROM positions WHERE encoded = ?", (encoded,)).fetchone()


def insert_or_replace_position(conn, encoded, fen, cp, policy, value, sample_count, legal_moves):
    # Insert or replace the position
    conn.execute("""
        INSERT OR REPLACE INTO positions (encoded, fen, cp, policy, value, sample_count, legal_moves)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT DO UPDATE SET
            fen = excluded.fen,
            cp = excluded.cp,
            policy = excluded.policy,
            value = excluded.value,
            sample_count = excluded.sample_count
    """, (encoded, fen, cp, policy, value, sample_count, legal_moves))


def process_game(line, actionspace, db_dict, verbose=False):
    try:
        pgn = io.StringIO(line)
        game = chess.pgn.read_game(pgn)
        if not game:
            return
        if game.next() is None:
            return
        node = Node(1, game.board(), actionspace, None, None)
        # game = game.next()  # skip starting position
        while True:
            move_made = game.next().move
            if game.board().turn == chess.WHITE:
                fen = game.board().fen()
            else:
                # flip board such that the king is always on the right
                fen = game.board().mirror().fen()
                move_made = chess.Move(
                    chess.square_mirror(move_made.from_square),
                    chess.square_mirror(move_made.to_square),
                    move_made.promotion
                )

            cp = game.eval()
            if cp is None:
                cp = 0
            else:
                cp = cp.pov(chess.WHITE)
                cp = cp.score(mate_score=12800)
            value = map_centipawns_to_probability(cp)
            encoded = node.encoded.astype(np.int8).tobytes()

            # insert into db_dict
            if encoded in db_dict:
                db_row = db_dict[encoded]
                db_row[2][actionspace.get_key(move_made)] += 1
                db_row[4] += 1
                if abs(db_row[1]) > abs(cp):
                    db_row[1] = cp
                    db_row[3] = value
            else:
                policy = np.zeros(actionspace.size, dtype=np.uint32)
                policy[actionspace.get_key(move_made)] += 1
                db_dict[encoded] = [fen, cp, policy, value, 1, game.board().legal_moves.count()]

            # move to next position
            game = game.next()
            if not game:
                return
            if not game.eval():
                return
            # update the tree
            node.add_child(game.move)
            children = node.children
            node = children[actionspace.get_key(game.move)]
    except Exception as e:
        if verbose:
            print(e)
            print(line)


def parse_worker(shard_path, queue):
    db_dict = {}
    actionspace = asp()
    skip_game = False
    with open(shard_path) as file:
        for line in file:
            if line == "\n":
                continue
            if line.startswith("WhiteElo"):
                elo = int(line.split('"')[1])
                if elo < 2200:
                    skip_game = True
                    continue
            if line.startswith("BlackElo"):
                elo = int(line.split('"')[1])
                if elo < 2200:
                    skip_game = True
                    continue
            if line.startswith("1."):
                if skip_game:
                    skip_game = False
                    continue
                if '[%eval' in line:
                    process_game(line, actionspace, db_dict)
    for encoded, (fen, cp, policy, value, sample_count, legal_moves) in db_dict.items():
        queue.put((encoded, fen, cp, policy, value, sample_count, legal_moves))


def encode_a85(byte_rep):
    return base64.a85encode(byte_rep).decode()


def decode_a85(byte_rep):
    return np.frombuffer(base64.a85decode(byte_rep), dtype=np.uint32)


def prune_database(conn):
    # remove all rows with sample_count < 2
    conn.execute("DELETE FROM positions WHERE sample_count < 2")


def should_prune(db_path):
    # prune if db is larger than 1TB
    if os.path.getsize(db_path) > 1e12:
        return True
    return False


def db_writer(queue, db_path, verbose=False):
    db_dict = {}
    conn = sqlite3.connect(db_path)
    row_count = 0
    exit_flag = False
    while not exit_flag:
        # combine rows from workers in memory
        while len(db_dict) < 100000:
            try:
                q_item = queue.get()
                if q_item is None:
                    exit_flag = True
                    break
                row_count += 1
                encoded, fen, cp, policy, value, sample_count, legal_moves = q_item
                if encoded in db_dict:
                    db_row = db_dict[encoded]
                    # update distribution of current position with move played
                    db_row[2] += policy
                    db_row[4] += sample_count
                    if abs(db_row[1]) > abs(cp):
                        db_row[1] = cp
                        db_row[3] = value
                else:
                    db_dict[encoded] = [fen, cp, policy, value, sample_count, legal_moves]
            except Exception as e:
                if verbose:
                    print(e)
                    print(q_item)

        # write to db
        for encoded, (fen, cp, policy, value, sample_count, legal_moves) in db_dict.items():
            encoded = encode_a85(encoded)
            row = get_position(conn, encoded)
            if row:
                cp = min(row[2], cp)
                value = min(row[4], value)
                # prevent overflow
                if row[5] + sample_count < 2**32 - 1:
                    policy = decode_a85(row[3]) + policy
                    policy = encode_a85(policy)
                    sample_count = row[5] + sample_count
                else:
                    sample_count = 2**32 - 1
                    policy = row[3]
            else:
                policy = encode_a85(policy)
            insert_or_replace_position(
                conn, encoded, fen, cp, policy, value, sample_count, legal_moves)
        db_dict = {}
        conn.commit()
        if should_prune(db_path):
            prune_database(conn, db_path)
    conn.close()


def parse(shard_path) -> None:
    tic = time.time()
    db_path = 'C:/sqlite_chess_db/lichess2200.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            encoded BLOB PRIMARY KEY NOT NULL UNIQUE,
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
    files = files[:1500]    #process 1500 files
    queue = mp.Manager().Queue(10000)
    db_worker = mp.Process(target=db_writer, args=(queue, db_path))
    db_worker.start()
    p_size = mp.cpu_count()
    # p_size = 1
    with mp.Pool(processes=p_size) as pool:
        for file in files:
            if file.endswith(".pgn"):
                pool.apply_async(parse_worker, args=(
                    shard_path + file, queue))
        pool.close()
        pool.join()
    queue.put(None)
    db_worker.join()
    toc = time.time()
    print(f"Processed {len(files)} files in {toc - tic} seconds")


def convert_to_numpy(arr):
    return np.frombuffer(arr, dtype=np.float32).reshape(16, 8, 8)


def display():
    db_path = 'C:/sqlite_chess_db/lichess2200.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM positions where sample_count > legal_moves order by RANDOM() LIMIT 1000")
    rows = cursor.fetchall()
    print(len(rows))
    for row in rows:
        fen = row[1]
        print(fen)
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


if __name__ == '__main__':
    # display()
    parse('E:/lichess_shards/lichess_db_standard_rated_2023-9/')

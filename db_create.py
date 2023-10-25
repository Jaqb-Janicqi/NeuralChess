import sqlite3
import math
import chess
import chess.pgn
import numpy as np
import os
import tqdm
from multiprocessing import Pool
import time
import io
from mcts_node import Node
from db_dataloader import DataLoader
import stockfish
from cache_read_priority import Cache


def get_centipawns(prob):
    return int(111.714640912 * math.tan(1.5620688421 * prob))


def map_centipawns_to_probability(centipawns):
    return math.atan2(centipawns, 111.714640912) / 1.5620688421


def insert_or_replace_position(conn, fen, cp, prob, encoded):
    # Insert or replace the position, keeping the one with the closest 'cp' to 0
    conn.execute("""
        INSERT OR REPLACE INTO positions (fen, cp, prob, encoded)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(fen) DO UPDATE
        SET 
            cp = CASE
                WHEN excluded.cp >= 0 THEN
                    CASE
                        WHEN positions.cp < 0 THEN excluded.cp
                        ELSE MIN(positions.cp, excluded.cp)
                    END
                ELSE
                    CASE
                        WHEN positions.cp >= 0 THEN excluded.cp
                        ELSE MAX(positions.cp, excluded.cp)
                    END
            END,
            prob = CASE
                WHEN ABS(positions.prob) < ABS(excluded.prob) THEN positions.prob
                ELSE excluded.prob
            END
    """, (fen, cp, prob, encoded))


def insert_or_abort(conn, fen, cp, prob, encoded):
    # Insert or replace the position, keeping the one with the closest 'cp' to 0
    conn.execute("""
        INSERT OR ABORT INTO positions (fen, cp, prob, encoded)
        VALUES (?, ?, ?, ?)
    """, (fen, cp, prob, encoded))


def insert_or_nothing(conn, fen, cp, prob, encoded):
    # Insert or update nothing
    conn.execute("""
        INSERT OR IGNORE INTO positions (fen, cp, prob, encoded)
        VALUES (?, ?, ?, ?)
    """, (fen, cp, prob, encoded))


def get_position_count(conn):
    return conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]


def process_game_by_io(conn, line):
    try:
        pgn = io.StringIO(line)
        game = chess.pgn.read_game(pgn)
        if not game:
            return
        game = game.next()  # skip starting position
        while True:
            if not game:
                break
            if game.eval() is None:
                game = game.next()
                continue
            # get the fen
            fen = game.board().fen()
            node = Node(1, game.board(), {}, None, None)
            # get the evaluation
            centipawns = game.eval()
            centipawns = centipawns.relative.score(mate_score=12800)
            # get the probability from the evaluation
            prob_eval = map_centipawns_to_probability(centipawns)
            # insert or replace the position
            insert_or_nothing(conn, fen, centipawns,
                              prob_eval, node.encoded.tobytes())
            game = game.next()
    except Exception as e:
        print(e)
        print(line)


def shard_parser(shard_path) -> None:
    start = 84785579
    conn = sqlite3.connect('C:/sqlite_chess_db/lichess.db')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY,
            fen TEXT NOT NULL UNIQUE,
            cp INTEGER NOT NULL,
            prob REAL NOT NULL,
            encoded BLOB NOT NULL
        );
    """)
    conn.commit()

    shard_name = os.path.basename(shard_path)
    pbar = tqdm.tqdm(desc="Processing shard " +
                     shard_name + ".", dynamic_ncols=True)
    eval_games_count = 0
    with open(shard_path) as file:
        # select until "1." is found
        for line in file:
            pbar.update(1)
            if line.startswith("1."):
                # check if the position has an eval
                if '[%eval' in line:
                    process_game_by_io(conn, line)
                    eval_games_count += 1
                    pbar.set_postfix({"eval games": eval_games_count})
                    if eval_games_count % 100 == 0:
                        conn.commit()
        conn.commit()
        conn.close()


def check_encoded():
    conn = sqlite3.connect('C:/sqlite_chess_db/lichess.db')
    cursor = conn.cursor()
    max_id = cursor.execute(
        "SELECT MAX(id) FROM positions").fetchone()[0]
    min_id = cursor.execute(
        "SELECT MIN(id) FROM positions").fetchone()[0]
    replace_query = "INSERT OR REPLACE INTO positions (id, fen, cp, prob, encoded) VALUES (?, ?, ?, ?, ?)"
    pbar = tqdm.tqdm(dynamic_ncols=True)
    for i in range(min_id, max_id + 1):
        x = cursor.execute(
            "SELECT * FROM positions where id=? limit 1", (i,)).fetchall()
        for row in x:
            id, fen, cp, prob, encoded = row
            board = chess.Board(fen=fen)
            node = Node(1, board, {}, None, None)
            node_encoded = node.encoded.tobytes()
            try:
                assert encoded == node_encoded
            except:
                conn.execute(replace_query, (id, fen, cp, prob, node_encoded))
                conn.commit()
                pbar.update(1)
    conn.commit()


def convert_to_numpy(arr):
    return np.frombuffer(arr, dtype=np.float32).reshape(16, 8, 8)


def create_new(total_time):
    start_time = time.time()
    conn = sqlite3.connect('C:/sqlite_chess_db/stockfish.db')
    cursor = conn.cursor()
    stockfish_engine = stockfish.Stockfish(
        path="D:/stockryba/stockfish-windows-x86-64-avx2.exe",
        depth=20,
        parameters={
            'Threads': 4,
        }
    )
    print(stockfish_engine.get_parameters())
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY,
            fen TEXT NOT NULL,
            cp INTEGER NOT NULL,
            prob REAL NOT NULL,
            encoded BLOB NOT NULL
        )
    """)
    conn.commit()
    pos_count = 0
    pbar = tqdm.tqdm()
    # cache = Cache(6000)
    cache = {}

    db_size = conn.execute(
        "SELECT COUNT(*) FROM positions").fetchone()[0]

    # get all the positions one by one
    for i in range(1, db_size + 1):
        x = cursor.execute(
            "SELECT * FROM positions where id=? limit 1", (i,)).fetchall()
        for row in x:
            id, fen, cp, prob, encoded = row
            cache[fen] = True
            pos_count += 1
            pbar.update(1)

    while start_time + total_time > time.time():
        board = chess.Board()
        node = Node(1, board, {}, None, None)

        while not board.is_game_over(claim_draw=True):
            board.push(np.random.choice(list(board.legal_moves)))
            fen = board.fen()
            if fen in cache:
                continue
            if board.fullmove_number > 50:
                break
            cache[fen] = True
            stockfish_engine.set_fen_position(fen)
            cp = stockfish_engine.get_evaluation()
            if cp["type"] == "mate":
                if cp["value"] == 0:
                    cp = 12800 * np.sign(cp["value"])
                else:
                    cp = 12800 / ((abs(cp["value"]) + 1)
                                  * np.sign(cp["value"]))
            else:
                cp = cp["value"]
            prob = map_centipawns_to_probability(cp)
            encode_str = node.encoded.tobytes()
            insert_or_nothing(conn, fen, cp, prob, encode_str)
            pos_count += 1
            pbar.update(1)
            if pos_count % 100 == 0:
                conn.commit()
    conn.commit()
    conn.close()


if __name__ == '__main__':
    create_new(10 * 60*60)
    # shard_parser('E:/chess_db/shard_2023-05.pgn')
    # shard_parser('E:/chess_db/shard_2023-06.pgn')

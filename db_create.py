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
from db_dataloader import db_dataloader
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


def get_position_count(conn):
    return conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]


def safe_commit(conn):
    for retry in range(10):
        try:
            conn.commit()
            return
        except sqlite3.OperationalError:
            time.sleep(5)


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
            # get the evaluation
            centipawns = game.eval()
            centipawns = centipawns.relative.score(mate_score=12800)
            # get the probability from the evaluation
            prob_eval = map_centipawns_to_probability(centipawns)
            # insert or replace the position
            insert_or_replace_position(conn, fen, centipawns, prob_eval)
            game = game.next()
    except Exception as e:
        print(e)
        print(line)


def shard_parser(shard_path) -> None:
    conn = sqlite3.connect('C:/sqlite_chess_db/chess_positions.db')
    shard_name = os.path.basename(shard_path)
    pbar = tqdm.tqdm(total=100000000,
                     desc="Processing shard " + shard_name + ".")
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
                if eval_games_count % 500 == 0:
                    conn.commit()
        conn.commit()
        conn.close()

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


def db_parser():
    conn = sqlite3.connect('C:/sqlite_chess_db/chess_positions.db')
    table_name = "positions"
    cursor = conn.cursor()
    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("array", convert_array)
    # select all rows where "encoded" is not null
    id, fen, cp, prob, encoded = cursor.execute(
        "SELECT * FROM {}".format(table_name)).fetchone()
    encoded = convert_array(encoded)
    print(encoded.shape)
    
    
    id_count = get_position_count(conn)
    for idx in tqdm.trange(id_count):
        idx = idx + 1
        id, fen, cp, prob, encoded = cursor.execute(
            "SELECT * FROM {} WHERE id = {}".format(table_name, idx)).fetchone()
        # x = conn.execute(
        #         f"SELECT * FROM positions LIMIT {10} OFFSET {0}")
        board = chess.Board(fen)
        node = Node(1, board, {}, None, None)
        insert_or_replace_position(conn, fen, cp, prob, node.encoded)
        if idx % 500 == 0:
            conn.commit()
    conn.commit()



def create_new(total_time):
    start_pos = chess.Board()
    start_time = time.time()
    conn = sqlite3.connect('C:/sqlite_chess_db/chess_positions.db')
    cursor = conn.cursor()
    stockfish_engine = stockfish.Stockfish(
        path="D:/stockryba/stockfish-windows-x86-64-avx2.exe",
        depth=20,
        parameters={
            'Threads': 4,
        }
    )
    print(stockfish_engine.get_parameters())
    # create the table
    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("array", convert_array)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY,
            fen TEXT NOT NULL,
            cp INTEGER NOT NULL,
            prob REAL NOT NULL,
            encoded array NOT NULL
        )
    """)
    conn.commit()
    pos_count = 0
    pbar = tqdm.tqdm()
    cache = Cache(8192)
    
    while start_time + total_time > time.time():
        # play a game
        board = chess.Board()
        node = Node(1, board, {}, None, None)
        while not board.is_game_over(claim_draw=True):
            board.push(np.random.choice(list(board.legal_moves)))
            fen = board.fen()
            if fen in cache:
                continue
            cache.add(fen, True)
            stockfish_engine.set_fen_position(fen)
            cp = stockfish_engine.get_evaluation()
            if cp["type"] == "mate":
                if cp["value"] == 0:
                    cp = 12800 * np.sign(cp["value"])
                else:
                    cp = 12800 / ((abs(cp["value"]) + 1) * np.sign(cp["value"]))
            else:
                cp = cp["value"]
            prob = map_centipawns_to_probability(cp)
            insert_or_abort(conn, fen, cp, prob, node.encoded)
            pos_count += 1
            pbar.update(1)
            if pos_count % 100 == 0:
                conn.commit()
    conn.commit()


if __name__ == '__main__':
    create_new(16 * 60*60)

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



def get_centipawns(prob):
    return int(111.714640912 * math.tan(1.5620688421 * prob))


def map_centipawns_to_probability(centipawns):
    return math.atan2(centipawns, 111.714640912) / 1.5620688421


def insert_or_replace_position(conn, fen, cp, prob):
    # Insert or replace the position, keeping the one with the closest 'cp' to 0
    conn.execute("""
        INSERT OR REPLACE INTO positions (fen, cp, prob)
        VALUES (?, ?, ?)
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
    """, (fen, cp, prob))

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


if __name__ == '__main__':
    timer_start = time.time()
    # shards_dir = "C:/Users/janic/Documents"
    shards_dir = "C:/Users/janic/Downloads"
    shards = os.listdir(shards_dir)
    paths = []
    for file_name in shards:
        file_name_base, file_extension = os.path.splitext(file_name)
        if file_extension != ".pgn":
            continue
        folder_path = os.path.join(shards_dir, file_name)
        paths.append(folder_path)
    for path in paths:
        shard_parser(path)

    # with Pool(4) as pool:
    #     pool.map(shard_parser, args)
    #     pool.close()
    #     pool.join()

    timer_end = time.time()
    # convert to minutes or hours if necessary
    if timer_end - timer_start > 60:
        if timer_end - timer_start > 3600:
            print("Processed " + str(len(paths)) + " files" + " in " +
                  str((timer_end - timer_start) / 3600) + " hours")
        else:
            print("Processed " + str(len(paths)) + " files" + " in " +
                  str((timer_end - timer_start) / 60) + " minutes")

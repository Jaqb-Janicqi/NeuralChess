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


def process_game_by_io(conn, line):
    try:
        pgn = io.StringIO(line)
        game = chess.pgn.read_game(pgn)

        db_accessed = False
        while True:
            # read the game and quit if file has ended
            game = chess.pgn.read_game(pgn)
            if not game:
                break
            game = game.next()  # skip starting position
            if not game:
                break
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
                db_accessed = True
                game = game.next()
        if db_accessed:
            conn.commit()
    except Exception as e:
        print(e)
        print(line)


def shard_parser(shard_path) -> None:
    eval_games_count = 0
    conn = sqlite3.connect('C:\sqlite_chess_db\chess_positions.db')
    pbar = tqdm.tqdm(leave=True)
    shard_name = os.path.basename(shard_path)
    with open(shard_path) as file:
        # select until "1." is found
        for line in file:
            if line.startswith("1."):
                # check if the position has an eval
                if '[%eval' in line:
                    process_game_by_io(conn, line)
                    eval_games_count += 1
                    pbar.set_description(
                        "Processing shard " + shard_name + ". Eval games found: " + str(eval_games_count) + ".")
                pbar.update(1)


if __name__ == '__main__':
    timer_start = time.time()
    conn = sqlite3.connect('C:\sqlite_chess_db\chess_positions.db')
    shards_dir = "E:\chess_db"
    shards = os.listdir(shards_dir)
    paths = []
    for file_name in shards:
        if ".pgn" not in file_name:
            continue
        folder_path = os.path.join(shards_dir, file_name)
        paths.append(folder_path)
    # for path in dirs:
    #     process_shards_in_path(path)

    with Pool(5) as pool:
        pool.map(shard_parser, paths)
        pool.close()
        pool.join()
    conn.commit()
    conn.close()

    timer_end = time.time()
    # convert to minutes or hours if necessary
    if timer_end - timer_start > 60:
        if timer_end - timer_start > 3600:
            print("Processed " + str(len(paths)) + " files" + " in " +
                  str((timer_end - timer_start) / 3600) + " hours")
        else:
            print("Processed " + str(len(paths)) + " files" + " in " +
                  str((timer_end - timer_start) / 60) + " minutes")

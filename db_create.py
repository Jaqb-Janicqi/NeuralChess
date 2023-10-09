import sqlite3
import math
import chess
import chess.pgn
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool

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

def process_shards_in_path(path):
    conn = sqlite3.connect('C:\sqlite_chess_db\chess_positions.db')
    shards = os.listdir(path)
    for shard in shards:
        shard_path = os.path.join(path, shard)
        # load pgn file
        pgn = open(shard_path)
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


if __name__ == '__main__':
    conn = sqlite3.connect('C:\sqlite_chess_db\chess_positions.db')
    folders_dir = "E:\chess_db"
    shard_folders = os.listdir(folders_dir)
    dirs = []
    for folder in shard_folders:
        if ".pgn" in folder:
            continue
        folder_path = os.path.join(folders_dir, folder)
        dirs.append(folder_path)
    # for path in dirs:
    #     process_shards_in_path(path)

    with Pool(5) as pool:
        pool.map(process_shards_in_path, dirs)
        pool.close()
        pool.join()
    conn.commit()
    conn.close()


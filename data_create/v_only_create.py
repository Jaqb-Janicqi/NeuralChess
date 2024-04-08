import sqlite3
import chess
import chess.pgn
import numpy as np
import os
import tqdm
import time
import io


import multiprocessing as mp
from collections import OrderedDict
import psutil

UINT32_MAX = 2**32 - 1
MAX_MEM_USAGE = 10*1024*1024*1024   # xGB
DUMP_SIZE = int(1e2)
DUMP_QUEUE_SIZE = int(1e6)
COMMIT_SIZE = int(1e4)
FILE_START = 0
# FILE_END = None
FILE_END = 2000
DBPATH = 'C:/sqlite_chess_db/lichess_evals.db'


def process_game(line, db_dict: dict, verbose: bool):
    try:
        pgn = io.StringIO(line)
        game = chess.pgn.read_game(pgn)
        if not game:
            return
        if game.next() is None:
            return
        while True:
            fen = game.board().fen()
            cp = game.eval()
            if cp is None:
                cp = 0
            else:
                cp = cp.relative.score(mate_score=12800)
            if fen in db_dict:
                pass
            else:
                num_pieces = len(game.board().piece_map())
                db_dict[fen] = (cp, num_pieces)

            # move to next position
            game = game.next()
            if not game:
                return
            if not game.eval():
                return
    except Exception as e:
        if verbose:
            print(e)
            print(line)


def parse_worker(shard_path, queue: mp.Queue):
    db_dict = OrderedDict()
    skip_game = False
    with open(shard_path) as file:
        for line in file:
            if line == '\n':
                continue

            if line.startswith("1."):
                if skip_game:
                    skip_game = False
                    continue
                if '[%eval' in line:
                    process_game(line, db_dict, verbose=True)

    # pass db_dict to db_queue
    queue.put(db_dict)


def write_worker(queue, db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    written = 0
    while True:
        item = queue.get()
        if item is None:
            break
        fen, cp, num_pieces = item
        command_str = f"INSERT INTO positions_{num_pieces} VALUES (?, ?)"
        c.execute(command_str, (fen, cp))
        written += 1
        if written % COMMIT_SIZE == 0:
            conn.commit()
            written = 0
    conn.commit()


def data_worker(parser_queue: mp.Queue, db_queue: mp.Queue):
    db_dict = OrderedDict()
    p_id = mp.current_process().pid

    while True:
        mem_usage = psutil.Process(p_id).memory_info().vms
        # start writing to db if mem usage is too high
        if mem_usage > MAX_MEM_USAGE:
            for _ in range(DUMP_SIZE):
                try:
                    fen, (cp, num_pieces) = db_dict.popitem(last=False)
                except KeyError:
                    break
                db_queue.put((fen, cp, num_pieces))
            continue

        # wait for item from parser
        while parser_queue.empty():
            time.sleep(1)

        # get item from parser
        q_dict = parser_queue.get()
        # exit if parsing has finished
        if q_dict is None:
            break

        # merge dictionaries
        db_dict.update(q_dict)

    # dump remaining items
    for _ in range(len(db_dict)):
        fen, (cp, num_pieces) = db_dict.popitem(last=False)
        db_queue.put((fen, cp, num_pieces))
    db_queue.put(None)


def parse(shard_path) -> None:
    tic = time.time()
    conn = sqlite3.connect(DBPATH)
    cursor = conn.cursor()
    for num_pieces in range(2, 33):
        cursor.execute(
            f"CREATE TABLE IF NOT EXISTS positions_{num_pieces} (fen text, cp int)")
    conn.commit()
    conn.close()

    files = os.listdir(shard_path)
    if FILE_END is not None:
        file_end = FILE_END
    else:
        file_end = len(files)
    files = files[FILE_START:file_end]

    # aggregate data from parsers to data_worker, concurrent writes to db
    with mp.Manager() as manager:
        parser_queue = manager.Queue(maxsize=DUMP_QUEUE_SIZE)
        db_queue = manager.Queue(maxsize=DUMP_QUEUE_SIZE)
        data_aggregator = mp.Process(
            target=data_worker, args=(parser_queue, db_queue))
        data_aggregator.start()

        # start write worker
        p_size = mp.cpu_count()
        writer = mp.Process(
            target=write_worker, args=(db_queue, DBPATH))
        writer.start()

        # start parse workers
        with mp.Pool(processes=p_size) as pool:
            pbar = tqdm.tqdm(total=len(files), desc="Parsing files",
                             leave=False, dynamic_ncols=True, position=0)
            for file in files:
                if file.endswith(".pgn"):
                    pool.apply_async(parse_worker, args=(
                        shard_path + file, parser_queue), callback=lambda _: pbar.update())
            pool.close()
            pool.join()
        pbar.close()
        parser_queue.put(None)
        data_aggregator.join()
        db_queue.put(None)
        writer.join()        
        toc = time.time()
        print(f"Processed {len(files)} files in {(toc - tic)/60} minutes")


if __name__ == '__main__':
    parse('E:/lichess_shards/lichess_db_standard_rated_2023-9/')
    # parse('E:/lichess_shards/lichess_db_standard_rated_2023-10/')

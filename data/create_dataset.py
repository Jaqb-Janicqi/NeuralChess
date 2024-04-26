import math
import chess
import chess.pgn
import os
import pandas as pd
import tqdm
import time
import io
import numpy as np

import multiprocessing as mp
import psutil


UINT32_MAX = 2**32 - 1
MAX_MEM_USAGE = 22*1024*1024*1024   # xGB
PARSER_QUEUE_SIZE = int(32)
DATAFRAME_BUFFER_SIZE = int(4)
FILE_START = 0
FILE_END = 1


def get_centipawns(prob):
    return int(111.714640912 * math.tan(1.5620688421 * prob))


def map_centipawns_to_probability(centipawns):
    return math.atan2(centipawns, 111.714640912) / 1.5620688421


def process_game(line, data, silent=False):
    try:
        pgn = io.StringIO(line)
        game = chess.pgn.read_game(pgn)
        if not game:
            return
        if game.next() is None:
            return
        while True:
            fen = game.board().fen()
            if fen not in data:
                eval = game.eval()

                if eval is None:
                    cp = 0
                    mate = None
                else:
                    eval = eval.white()
                    cp = eval.score(mate_score=12800)
                    mate = str(eval.mate()) if eval.is_mate() else None
                win_prob = map_centipawns_to_probability(cp)

                ply = game.board().ply()
                data[fen] = (cp, win_prob, mate, ply)

            # move to next position
            game = game.next()
            if not game:
                return
            if not game.eval():
                return
            if not game.next():
                return
    except Exception as e:
        if not silent:
            print(e)
            print(line)


def parse_worker(shard_path, parser_queue: mp.Queue):
    data = {}
    with open(shard_path) as file:
        for line in file:
            if line == '\n':
                continue
            if line.startswith("1."):
                if '[%eval' in line:
                    process_game(line, data)

    df = pd.DataFrame.from_dict(data, orient='index').reset_index()
    df.columns = ['fen', 'cp', 'win_prob', 'mate', 'ply']
    parser_queue.put(df)


def save_csv(df, path):
    df.to_csv(path, index=False)


def concat_dfs(dfs):
    concatenated = pd.concat(dfs, ignore_index=True)
    concatenated.drop_duplicates(subset=['fen'], keep='first', inplace=True)
    return concatenated


def data_worker(parser_queue: mp.Queue) -> None:
    dataframes = []
    p_id = mp.current_process().pid
    last_df = 0

    while True:
        mem_usage = psutil.Process(p_id).memory_info().vms
        # concat dataframes if memory usage is too high
        if mem_usage > MAX_MEM_USAGE:
            dataframes = [concat_dfs(dataframes)]
            time.sleep(10)
            mem_usage = psutil.Process(p_id).memory_info().vms
            if mem_usage > MAX_MEM_USAGE:
                break

        # wait for item from parser
        while parser_queue.empty():
            time.sleep(0.1)

        # get item from parser
        q_item = parser_queue.get()
        # exit if parsing has finished
        if q_item is None:
            break

        # add item to dataframes if there is space
        if len(dataframes) < DATAFRAME_BUFFER_SIZE:
            dataframes.append(q_item)
            continue

        # add item to one of existing dataframes
        dataframes[(last_df) % len(dataframes)] = pd.concat(
            [dataframes[last_df], q_item], ignore_index=True)
        last_df = (last_df+1) % len(dataframes)

    # save to csv
    df = concat_dfs(dataframes)
    save_csv(df, f'./positions.csv')


def parse(shard_path) -> None:
    tic = time.time()
    files = os.listdir(shard_path)

    p_size = mp.cpu_count() - 2

    if FILE_END is not None:
        file_end = FILE_END
    else:
        file_end = len(files)
    files = files[FILE_START:file_end]

    parser_queue = mp.Manager().Queue(PARSER_QUEUE_SIZE)
    data_agregator = mp.Process(target=data_worker, args=(parser_queue,))
    data_agregator.start()

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
    data_agregator.join()
    toc = time.time()
    print(f"Processed {len(files)} files in {(toc - tic)/60} minutes")


def generate_positions(board, depth):
    positions = [board.fen()]
    if depth == 0:
        return positions

    legal_moves = list(board.legal_moves)
    for move in legal_moves:
        board.push(move)
        positions.extend(generate_positions(board, depth - 1))
        board.pop()

    return positions


def generate_random_at_depth(board, depth):
    if depth == 0:
        return board.fen()

    legal_moves = list(board.legal_moves)
    move = np.random.choice(legal_moves)
    board.push(move)
    while board.is_checkmate() or board.is_stalemate():
        board.pop()
        move = np.random.choice(legal_moves)
        board.push(move)
    return generate_random_at_depth(board, depth - 1)


def generate_random_to_depth(board, start_depth, end_depth):
    positions = [board.fen()]
    if start_depth == end_depth:
        return positions

    legal_moves = list(board.legal_moves)
    move = np.random.choice(legal_moves)
    board.push(move)
    while board.is_checkmate() or board.is_stalemate():
        board.pop()
        move = np.random.choice(legal_moves)
        board.push(move)
    positions.extend(generate_random_to_depth(board, start_depth + 1, end_depth))
    return positions


def create_stockfish_evals():
    eval_time = 200
    positions = []
    positions.extend(generate_positions(chess.Board(), 4))
    for _ in range(10000):
        positions.extend(generate_random_to_depth(chess.Board(), 0, 12))
    for pos in positions:
        # check if position is str
        if not isinstance(pos, str):
            positions.remove(pos)
    positions = pd.DataFrame(positions, columns=['fen'])
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    # drop duplicates
    positions = positions[~positions['fen'].isin(df_train['fen'])]
    positions = positions[~positions['fen'].isin(df_test['fen'])]
    del df_train
    del df_test
    positions = positions['fen'].tolist()
    print(f"Generated {len(positions)} positions")

    evals = []
    with chess.engine.SimpleEngine.popen_uci("C:/Users/janic/Desktop/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe") as engine:
        engine.configure({"Threads": 16, "Hash": 1024*24, "Skill Level": 20})

        for fen in tqdm.tqdm(positions, desc="Evaluating positions", leave=False, dynamic_ncols=True):
            board = chess.Board(fen)

            info = engine.analyse(board, chess.engine.Limit(time=eval_time/1000))

            ply = board.ply()
            eval = info["score"].white()
            cp = eval.score(mate_score=12800)
            mate = str(eval.mate()) if eval.is_mate() else None
            win_prob = map_centipawns_to_probability(cp)
            evals.append((fen, cp, win_prob, mate, ply))

    df = pd.DataFrame(evals, columns=['fen', 'cp', 'win_prob', 'mate', 'ply'])
    df.to_csv(f'stockfish_{eval_time}ms.csv', index=False)


def train_test_split():
    df = pd.read_csv('data/positions_full.csv')
    
    # filter df to only include positions with ply lower than 20
    df_low_ply = df[df['ply'] <= 20].copy()
    df.drop(df_low_ply.index, inplace=True)

    df_high_ply = df[df['ply'] > 20].loc[:len(df_low_ply)].copy()
    df.drop(df_high_ply.index, inplace=True)

    test_low_ply = df_low_ply.sample(frac=0.1)
    df_low_ply.drop(test_low_ply.index, inplace=True)

    train = pd.concat([df_low_ply, df_high_ply])
    train = train.drop_duplicates(subset=['fen'], keep='first')
    train.to_csv('train.csv', index=False)

    df_test_sample = df.sample(n=5000000)
    test = pd.concat([df_test_sample, test_low_ply])
    test = test.drop_duplicates(subset=['fen'], keep='first')
    test.to_csv('test.csv', index=False)


if __name__ == '__main__':
    train_test_split()
    # create_stockfish_evals()
    # parse('E:/lichess_shards/lichess_db_standard_rated_2023-9/')
    # parse('E:/lichess_shards/lichess_db_standard_rated_2023-10/')

import os
import tqdm
import time

timer_start = time.time()
shard_num = 0
file_num = 0
eval_file_num = 0
shards_path = "E:\chess_db"
pbar = tqdm.tqdm()

shards = os.listdir(shards_path)
for shard in shards:
    path = os.path.join("E:\chess_db", shard)
    pbar.reset()
    pbar.set_description("Processing shard " + shard)
    shard_name, month = shard.split("-")
    month = month.split(".")[0]
    shard_name, shard_year = shard_name.split("_")
    folder_name = shard_name + "_" + shard_year + "_" + month
    folder_path = os.path.join(shards_path, folder_name)
    with open(path) as file:
        # select until "1." is found
        position = []
        for line in file:
            position.append(line)
            if position[-1].startswith("1."):
                # check if the position has an eval
                if '[%eval' in position[-1]:
                    # write the position to a new file
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    # remove first empty line
                    if position[0] == "\n":
                        position.pop(0)
                    # write the position to a new file
                    with open(os.path.join(folder_path, "game" + str(eval_file_num) + ".pgn"), "w") as new_file:
                        for line in position:
                            new_file.write(line)
                    eval_file_num += 1
                position = []
                file_num += 1
                pbar.update(1)
    file_num = 0

timer_end = time.time()
# convert to minutes or hours if necessary
if timer_end - timer_start > 60:
    if timer_end - timer_start > 3600:
        print("Processed " + str(file_num) + " files" + " in " +
              str((timer_end - timer_start) / 3600) + " hours")
    else:
        print("Processed " + str(file_num) + " files" + " in " +
              str((timer_end - timer_start) / 60) + " minutes")
print("Found " + str(eval_file_num) + " files with evals")

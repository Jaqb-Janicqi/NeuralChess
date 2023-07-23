import numpy as np

Move = np.dtype([
    ("src_y", np.ubyte),
    ("src_x", np.ubyte),
    ("dst_y", np.ubyte),
    ("dst_x", np.ubyte),
    ("promo_piece", np.byte),
    ("special_dir", np.byte)
])

import numpy as np

#float to int
def compress(row):
    max_val=np.max(row)
    min_val=np.min(row)
    compressed=np.zeros_like(row, dtype=np.uint8)
    
    max_idx=np.argmax(row)
    
    if max_val==min_val:
        compressed[:]=0
        compressed[max_idx]=255
    else:
        for i, val in enumerate(row):
            if i==max_idx:
                compressed[i]=255
            else:
                compressed[i]=int(254*(val-min_val)/(max_val-min_val))
    
    return compressed

#int to float
def decompress(row, min_val, max_val):
    max_idx=np.argmax(row)
    
    if max_val==min_val:
        return np.full_like(row, max_val, dtype=np.float32)
    
    decompressed=np.zeros_like(row, dtype=np.float32)
    
    for i, val in enumerate(row):
        if i==max_idx:
            decompressed[i]=max_val
        else:
            decompressed[i]=(val/254.0)*(max_val-min_val)+min_val
    
    return decompressed

matrix_float=np.array([
    [1.2, 3.5, 2.1, 0.5],
    [4.0, 1.8, 2.5, 3.2],
    [0.8, 0.9, 0.7, 0.6]
], dtype=np.float32)

#输出float to int
compressed_rows = []
min_vals = []
max_vals = []

for i in range(matrix_float.shape[0]):
    comp_row = compress(matrix_float[i])
    compressed_rows.append(comp_row)
    min_vals.append(np.min(matrix_float[i]))
    max_vals.append(np.max(matrix_float[i]))

compressed = np.array(compressed_rows)
print(compressed)

#输出int to float
decompressed_rows = []
for i in range(compressed.shape[0]):
    decomp_row = decompress(compressed[i], min_vals[i], max_vals[i])
    decompressed_rows.append(decomp_row)

decompressed = np.array(decompressed_rows)
print(decompressed)
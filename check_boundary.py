import numpy as np
W, H = 256, 256
blurred = np.fromfile('images/blurred.raw', dtype=np.uint8).reshape(H, W)
print("Border pixel ranges:")
print(f"  Top row:    min={blurred[0,:].min()}, max={blurred[0,:].max()}")
print(f"  Bottom row: min={blurred[H-1,:].min()}, max={blurred[H-1,:].max()}")
print(f"  Left col:   min={blurred[:,0].min()}, max={blurred[:,0].max()}")
print(f"  Right col:  min={blurred[:,W-1].min()}, max={blurred[:,W-1].max()}")
print(f"  Interior:   min={blurred[2:H-2,2:W-2].min()}, max={blurred[2:H-2,2:W-2].max()}")

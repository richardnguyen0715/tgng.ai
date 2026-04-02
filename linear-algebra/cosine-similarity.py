import numpy as np

def cosine_simi(a, b):
		a = np.array(a, dtype=np.float64)
		b = np.array(b, dtype=np.float64)
		
		norm_a = np.linalg.norm(a)
		norm_b = np.linalg.norm(b)
		
		if norm_a == 0 or norm_b == 0:
				return 0
			
		return np.dot(a, b) / (norm_a * norm_b)
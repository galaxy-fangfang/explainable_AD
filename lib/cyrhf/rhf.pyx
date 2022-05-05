# -*- coding: utf-8 -*-
# @Author: Andrian Putina
# @Date:   2020-12-03 14:41:45
# @Last Modified by:   Andrian Putina
# @Last Modified time: 2020-12-03 14:41:45
#cython: language_level=3

from __future__ import print_function

import cython, json
from libc.stdlib cimport malloc, free
from cython.view cimport array as cvarray
import numpy as np
from cython cimport view
from libc.math cimport log


from cpython cimport set
cimport numpy as np

ctypedef np.float64_t DTYPE_t

from numpy import float64 as DTYPE
# ctypedef np.npy_float32 DTYPE_t

cdef DTYPE_t PRECISION_THRESHOLD = 1.77e-4
cdef DTYPE_t MAX_FLOAT64 = 1.7976931348623157e+308

"""
#######
STRUCTS
#######
"""

"""
Node Record for tree builder
"""
cdef struct NodeRecord:
	int start
	int end
	int depth
	bint is_leaf

"""
Node Record for decremental Kurtosis
"""
cdef struct NodeRecordKurt:
	int start
	int end
	int depth
	bint is_leaf
	int nodeID

"""
Incremental Kurtosis struct
"""
cdef struct INC_Kurtosis:
	int dimensionID
	DTYPE_t n, M1, M2, M3, M4, kvalue

"""
Splitter struct
"""
cdef struct Splitter:
	int attribute
	DTYPE_t threshold
	int position

"""
Min max attribute struct
"""
cdef struct min_max_struct:
	DTYPE_t min_
	DTYPE_t max_

"""
Incremental Kurtosis struct
"""
cdef struct kurtosis_struct:
	int dimensionID
	DTYPE_t n, M1, M2, M3, M4, kvalue

cdef bint debug = False

"""
#######
Classes
#######
"""
cdef class StackKurt:
	"""
	Stack used when Decremental Kurtosis is used in tree building
	"""
	cdef int size
	cdef int top
	cdef NodeRecordKurt *pointer

	def __cinit__(self):
		"""
		Hard coded to 15 as max heigh not > 8
		"""
		self.size = 15
		self.top = 0
		self.pointer = <NodeRecordKurt*> malloc(self.size * sizeof(NodeRecordKurt))

	def __dealloc__(self):
		free(self.pointer)

	cdef bint is_empty(self):
		return self.top <= 0

	cdef int push(self, int start, int end, int depth, bint is_leaf, int nodeID):
		"""
		Push element into Stack
		"""
		cdef int top = self.top
		cdef NodeRecordKurt *element = NULL

		element = self.pointer
		element[top].start = start
		element[top].end = end
		element[top].depth = depth
		element[top].is_leaf = is_leaf
		element[top].nodeID = nodeID

		self.top = top + 1
		return 0

	cdef int pop(self, NodeRecordKurt *element):
		"""
		Pop element from Stack
		"""
		cdef int top = self.top
		cdef NodeRecordKurt *stack = self.pointer

		if top <= 0:
			return -1

		element[0] = stack[top - 1]
		self.top = top - 1

		return 0

cdef class Stack:
	"""
	Stack used in tree building
	"""
	cdef int size
	cdef int top
	cdef NodeRecord *pointer

	def __cinit__(self):
		"""
		Hard coded to 15 as max heigh not > 8
		"""
		self.size = 15
		self.top = 0
		self.pointer = <NodeRecord*> malloc(self.size * sizeof(NodeRecord))

	def __dealloc__(self):
		free(self.pointer)

	cdef bint is_empty(self):
		return self.top <= 0

	cdef int push(self, int start, int end, int depth, bint is_leaf):
		"""
		Push element into Stack
		"""
		cdef int top = self.top
		cdef NodeRecord *element = NULL

		element = self.pointer
		element[top].start = start
		element[top].end = end
		element[top].depth = depth
		element[top].is_leaf = is_leaf

		self.top = top + 1
		return 0

	cdef int pop(self, NodeRecord *element):
		"""
		Pop element from Stack
		"""
		cdef int top = self.top
		cdef NodeRecord *stack = self.pointer

		if top <= 0:
			return -1

		element[0] = stack[top - 1]
		self.top = top - 1

		return 0

"""
#########
FUNCTIONS
#########
"""

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
cdef int* initializeArangeArray(int n):
	"""
	Init range array of size n
	"""
	cdef int *array 
	cdef int i
	array = <int *> malloc(n * sizeof(int))

	for i in range(n):
		array[i] = i

	return array


# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef min_max_struct minmax(DTYPE_t[:,::1] matrix, int start, int end, int dimensionID, int *indices):
	"""
	Get Min and Max in dimensionID
	"""
	cdef min_max_struct min_max_s
	cdef int i
	cdef DTYPE_t value

	min_max_s.min_ = matrix[indices[start],dimensionID]
	min_max_s.max_ = matrix[indices[start],dimensionID]

	for i from start <= i < end:
		value = matrix[indices[i], dimensionID]
		if value < min_max_s.min_:
			min_max_s.min_ = value
		if value > min_max_s.max_:
			min_max_s.max_ = value

	return min_max_s

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
cdef double compute_INC_Kurtosis(DTYPE_t[:,::1] matrix, int dimensionID, int start, int end, int *indices):
	"""
	Compute Kurtosis incrementally
	"""
	cdef DTYPE_t n1=0, delta=0, delta_n=0, delta_n2=0, term1=0, n=0, kvalue, M1=0, M2=0, M3=0, M4=0
	cdef int i 
	
	for i from start <= i < end: 
		n1 = n
		n += 1
		delta = matrix[indices[i], dimensionID] - M1
		delta_n = delta/n
		delta_n2 = delta_n * delta_n
		term1 = delta * delta_n * n1
		M1 = M1 + delta_n
		M4 = M4 + term1 * delta_n2 * (n*n - 3 * n + 3) + 6 * delta_n2 * M2 - 4 * delta_n * M3
		M3 = M3 + term1 * delta_n * (n - 2) - 3 * delta_n * M2
		M2 = M2 + term1


		if M4 > MAX_FLOAT64:
			print('OVERFLOW MAX_FLOAT64')

	"""
	Check if array is constant by checking VARIANCE
	"""
	if M2 > PRECISION_THRESHOLD:
		kvalue = (n * M4) / (M2 * M2)
		return kvalue
	else:
		return 0


# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef void partition(DTYPE_t[:,::1] matrix, int start, int end, int *indices, Splitter *splitter):
	"""
	Partition data
	Changes indices between 'start' and 'end' as to fit with splitter criterion
	"""
	cdef int i=start, j=end-1, tmp

	while i<=j:

		if matrix[indices[i], splitter.attribute] < splitter.threshold:
			# print('Entering if < threshold: i {}, indices {}, value={}'.format(i, indices[i], matrix[indices[i],splitter.attribute]))
			# print('Current status: i={} j={}'.format(i, j))
			i+=1
			# print('New status: i={} j={}'.format(i, j))
			# print()
		else:
			# print('Entering else, status: i={} j={}, value={}'.format(i, j, matrix[indices[j], splitter.attribute]))
			while (matrix[indices[j], splitter.attribute] >= splitter.threshold) and (j>i):
				# print('Entering while: j={}, value={}'.format(j, matrix[indices[j],splitter.attribute]))
				j -= 1

			if j<=i:
				break    

			tmp = indices[i]
			indices[i] = indices[j]
			indices[j] = tmp
			# print('Swapping: {} - {} | {} vs {}'.format(i,j, matrix[indices[i],splitter.attribute], matrix[indices[j],splitter.attribute]))
			i+=1
			# print()

	# print('Position : {}'.format(i))
	splitter.position = i

	# if (i-start == 1):
	# 	print('i==start : {} - {} - {} - {}'.format(i, start, end, splitter.threshold))
	# 	for x in range(start, end):
	# 		print(matrix[indices[x], splitter.attribute])

	if debug:
		print('Splitter Position: {}'.format(splitter.position))

	if i == start:
		print('PROBLEMMMM')
		print('i==start : i:{} - start:{} - end:{} - threshold:{} - diff: {}'.format(i, start, end, splitter.threshold, end-start))
		for x in range(start, end):
			print(matrix[indices[x], splitter.attribute])

		splitter.attribute = -1
		splitter.threshold = -1
		splitter.position = -1

cdef void update_kurtosis_attribute(DTYPE_t[:,::1] matrix, int start, position, int end, int *indices, kurtosis_struct *k_struct):
	"""
	Updates Kurtosis attribute decrementally
	"""
	cdef DTYPE_t n1, delta, delta_n, delta_n2, term1
	cdef int i
	cdef bint not_overflow = True
	# print('')

	if debug:
		print('\tUpdating DimensionID:{} - n: {}'.format(k_struct.dimensionID, k_struct.n))

	if position < (start+end)/2:
		index_min = start
		index_max = position
	else:
		index_min = position
		index_max = end

	if debug:
		print('\tRemoving from {} to {}'.format(index_min, index_max))

	
	# print('\tRemoving from {} to {}. Initial kvalue: {}'.format(start, end, k_struct.kvalue))
	# print('k_struct.n:{} - k_struct.k: {} - dimensionID: {}'.format(k_struct.n, k_struct.kvalue, k_struct.dimensionID))
	# print('Indices: {} - Matrix: {}'.format(indices[0], matrix[indices[0], k_struct.dimensionID]))

	for i from index_min <= i < index_max:
		n1 = k_struct.n
		k_struct.n -= 1
		delta = matrix[indices[i], k_struct.dimensionID] - k_struct.M1
		delta_n = delta/k_struct.n
		delta_n2 = delta_n * delta_n
		term1 = delta * delta_n * n1
		k_struct.M1 = k_struct.M1 - delta_n
		k_struct.M2 = k_struct.M2 - term1
		k_struct.M3 = k_struct.M3 - term1 * delta_n * (n1 - 2) + 3 * delta_n * k_struct.M2
		k_struct.M4 = k_struct.M4 - term1 * delta_n2 * (n1 * n1 - 3 * n1 + 3) - 6 * delta_n2 * k_struct.M2 + 4 * delta_n * k_struct.M3

		if k_struct.M4 > MAX_FLOAT64:
			print('OVERFLOW MAX_FLOAT64')

		if k_struct.M4 < 0:
			not_overflow = False
			break

	if not_overflow:
		if k_struct.M2 > PRECISION_THRESHOLD:
			k_struct.kvalue = (k_struct.n * k_struct.M4) / (k_struct.M2 * k_struct.M2)
		else:
			k_struct.kvalue = 0
	else:
		if debug:
			print('\tDetected overflow! Recompute!')
		if position < (start + end)/2:

			compute_INC_Kurtosis_struct(matrix, k_struct.dimensionID, position, end, indices, k_struct)
		else:
			compute_INC_Kurtosis_struct(matrix, k_struct.dimensionID, start, position, indices, k_struct)

	# if k_struct.M2 > PRECISION_THRESHOLD:
	# 	k_struct.kvalue = (k_struct.n * k_struct.M4) / (k_struct.M2 * k_struct.M2)
	# else:
	# 	k_struct.kvalue = 0

	if debug:
		print('\tUPDATED: n: {} - kvalue: {} - M2: {} - M4: {}'.format(k_struct.n, k_struct.kvalue, k_struct.M2, k_struct.M4))

cdef void copy_kstruct(kurtosis_struct origin_kstruct, kurtosis_struct *dest_kstruct):
	"""
	Copy kurtosis struct from origin to dest
	"""
	dest_kstruct.dimensionID = origin_kstruct.dimensionID
	dest_kstruct.n = origin_kstruct.n
	dest_kstruct.M1 = origin_kstruct.M1
	dest_kstruct.M2 = origin_kstruct.M2
	dest_kstruct.M3 = origin_kstruct.M3
	dest_kstruct.M4 = origin_kstruct.M4
	dest_kstruct.kvalue = origin_kstruct.kvalue

cdef void update_kurtosis(DTYPE_t[:,::1] matrix, int start, int position, int end, int d, int *indices, kurtosis_struct **nodes_kstruct, int nodeID, int nodeID2):
	"""
	Update Kurtosis left and right after split

	Copy kurtosis struct from parent = nodeID to child nodeID2
	Update nodeID2 struct
	"""

	cdef int i, j

	for i in range(d):
		copy_kstruct(nodes_kstruct[nodeID][i], &nodes_kstruct[nodeID2][i])
		update_kurtosis_attribute(matrix, start, position, end, indices, &nodes_kstruct[nodeID2][i])

	return

cdef void compute_INC_Kurtosis_struct(DTYPE_t[:,::1] matrix, int dimensionID, int start, int end, int *indices, kurtosis_struct *k_struct):
	"""
	Init struct and compute Kurtosis 
	"""
	cdef DTYPE_t n1=0, delta=0, delta_n=0, delta_n2=0, term1=0, n=0, M1=0, M2=0, M3=0, M4=0
	cdef int i 

	for i from start <= i < end: 
		n1 = n
		n += 1
		delta = matrix[indices[i], dimensionID] - M1
		delta_n = delta/n
		delta_n2 = delta_n * delta_n
		term1 = delta * delta_n * n1
		M1 = M1 + delta_n
		M4 = M4 + term1 * delta_n2 * (n*n - 3 * n + 3) + 6 * delta_n2 * M2 - 4 * delta_n * M3
		M3 = M3 + term1 * delta_n * (n - 2) - 3 * delta_n * M2
		M2 = M2 + term1


	k_struct.dimensionID = dimensionID
	k_struct.n = n
	k_struct.M1 = M1
	k_struct.M2 = M2
	k_struct.M3 = M3
	k_struct.M4 = M4

	"""
	Check if array is constant by checking VARIANCE
	"""
	if M2 > PRECISION_THRESHOLD:
		k_struct.kvalue = (n * M4) / (M2 * M2)
	else:
		k_struct.kvalue = 0

	if debug:
		print('k_struct.kvalue: {} - start:{} - end:{} - dimensionID: {} - n: {}'.format(k_struct.kvalue, start, end, k_struct.dimensionID, k_struct.n))

cdef void computeKurtosis(DTYPE_t[:,::1] matrix, int d, int start, int end, int *indices, kurtosis_struct *k_struct):
	cdef int i

	for i in range(d):
		compute_INC_Kurtosis_struct(matrix, i, start, end, indices, &k_struct[i])

	return

cdef kurtosis_struct** init_kstruct_array(int n_nodes, int d):
	"""
	Init Kstruct array for each node in the tree
	"""
	cdef kurtosis_struct **k_struct_array = <kurtosis_struct**>malloc(n_nodes * sizeof(kurtosis_struct*))
	cdef int i

	for i in range(n_nodes):
		k_struct_array[i] = <kurtosis_struct*> malloc(d * sizeof(kurtosis_struct))

	return k_struct_array

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
cdef void decremental_kurtosis_split(DTYPE_t[:,::1] matrix, int d, int *indices, int start, int end, Splitter *splitter, DTYPE_t[:] randoms, int *random_index, kurtosis_struct **nodes_kstruct, int nodeID, int n_nodes):
	"""
	Find kurtosis split and decrementally update kurtosis structs
	"""
	cdef int i, attribute
	cdef long double *Kurtosis_Values_log = <long double *> malloc(d * sizeof(long double))
	cdef long double Kurtosis_SUM = 0
	cdef double random_number
	cdef long double current_sum = 0, split_value = 0
	cdef min_max_struct min_max_s

	for i in range(d):
		Kurtosis_Values_log[i] = log(nodes_kstruct[nodeID][i].kvalue+1)
		Kurtosis_SUM += Kurtosis_Values_log[i]

	if debug:
		print('Kurtosis_SUM: {}'.format(Kurtosis_SUM))

	"""
	Impossible to SPLIT
	"""
	if Kurtosis_SUM == 0:
		splitter.attribute = -1
		splitter.threshold = -1
		splitter.position = -1
		return 

	"""
	Extract random number (from array of randoms) and select an attribute according to Kurtosis Split
	"""
	random_number = randoms[random_index[0]]*Kurtosis_SUM
	random_index[0] += 1

	for attribute in range(d):
		current_sum += Kurtosis_Values_log[attribute]
		if current_sum > random_number:
			break

	"""
	Compute min max of selected attribute and extract random split value
	"""
	min_max_s = minmax(matrix, start, end, attribute, indices)

	split_value = randoms[random_index[0]]*(min_max_s.max_ - min_max_s.min_) + min_max_s.min_
	random_index[0] += 1
	
	"""
	Splitter: assign attribute and threshold
	"""
	splitter.attribute = attribute
	splitter.threshold = split_value

	partition(matrix, start, end, indices, splitter)

	if splitter.position == -1:
		return

	"""
	Update kurtosis
	"""

	if splitter.position < (end+start)/2:

		"""
		LEFT SPLIT
		NodeID = Current + 1
		KURTOSIS TO RECOMPUTE on elements from start to splitter.position
		"""
		if debug:
			print('R-compute kurtosis left')
		computeKurtosis(matrix, d, start, splitter.position, indices, nodes_kstruct[n_nodes+1])

		"""
		RIGHT SPLIT
		NodeID = Current + 2
		KURTOSIS TO UPDATE removing elements from start to splitter.position
		"""
		if debug:
			print('Update kurtosis right')
		update_kurtosis(matrix, start, splitter.position, end, d, indices, nodes_kstruct, nodeID, n_nodes+2)



	else:
		"""
		LEFT SPLIT
		NodeID = Current + 1
		KURTOSIS TO UPDATE removing elements from splitter.position to end
		"""
		if debug:
			print('Update kurtosis left')
		update_kurtosis(matrix, start, splitter.position, end, d, indices, nodes_kstruct, nodeID, n_nodes+1)


		"""
		RIGHT SPLIT
		NodeID = Current + 2
		KURTOSIS TO RECOMPUTE on elements from splitter.position to end
		"""
		if debug:
			print('R-compute kurtosis right')		
		computeKurtosis(matrix, d, splitter.position, end, indices, nodes_kstruct[n_nodes+2])

	"""
	Free memory
	"""

	free(Kurtosis_Values_log)

	return

cdef void random_split(DTYPE_t[:,::1] matrix, int d, int *indices, int start, int end, Splitter *splitter, DTYPE_t[:] randoms, int *random_index):
	"""
	Splits data according to random split
	Used only to test against kurtosis split
	"""
	cdef int i, attribute
	cdef double *Kurtosis_Values_log = <double *> malloc(d * sizeof(double))
	cdef double Kurtosis_SUM = 0
	cdef double random_number, current_sum = 0, split_value = 0
	cdef min_max_struct min_max_s

	for i in range(d):
		if log(compute_INC_Kurtosis(matrix, i, start, end, indices)+1) > 0:
			Kurtosis_Values_log[i] = 1
		else:
			Kurtosis_Values_log[i] = 0
		Kurtosis_SUM += Kurtosis_Values_log[i]

	"""
	Impossible to SPLIT
	"""
	if Kurtosis_SUM == 0:
		splitter.attribute = -1
		splitter.threshold = -1
		splitter.position = -1
		return


	"""
	Extract random number (from array of randoms) and select an attribute according to Kurtosis Split
	"""
	random_number = randoms[random_index[0]]*Kurtosis_SUM
	random_index[0] += 1

	for attribute in range(d):
		current_sum += Kurtosis_Values_log[attribute]
		if current_sum > random_number:
			break

	"""
	Compute min max of selected attribute and extract random split value
	"""
	min_max_s = minmax(matrix, start, end, attribute, indices)

	split_value = randoms[random_index[0]]*(min_max_s.max_ - min_max_s.min_) + min_max_s.min_
	random_index[0] += 1
	
	"""
	Splitter: assign attribute and threshold
	"""
	splitter.attribute = attribute
	splitter.threshold = split_value

	"""
	Partition data with splitter
	"""
	partition(matrix, start, end, indices, splitter)

	free(Kurtosis_Values_log)

	return


cdef void kurtosis_split(DTYPE_t[:,::1] matrix, int d, int *indices, int start, int end, Splitter *splitter, DTYPE_t[:] randoms, int *random_index):
	"""
	Split data according to Kurtosis Split
	"""
	cdef int i, attribute
	cdef double *Kurtosis_Values_log = <double *> malloc(d * sizeof(double))
	cdef double Kurtosis_SUM = 0
	cdef double random_number, current_sum = 0, split_value = 0
	cdef min_max_struct min_max_s

	for i in range(d):
		Kurtosis_Values_log[i] = log(compute_INC_Kurtosis(matrix, i, start, end, indices)+1)
		Kurtosis_SUM += Kurtosis_Values_log[i]

	"""
	Impossible to SPLIT
	"""
	if Kurtosis_SUM == 0:
		splitter.attribute = -1
		splitter.threshold = -1
		splitter.position = -1
		return


	"""
	Extract random number (from array of randoms) and select an attribute according to Kurtosis Split
	"""
	random_number = randoms[random_index[0]]*Kurtosis_SUM
	random_index[0] += 1

	for attribute in range(d):
		current_sum += Kurtosis_Values_log[attribute]
		if current_sum > random_number:
			break

	"""
	Compute min max of selected attribute and extract random split value
	"""
	min_max_s = minmax(matrix, start, end, attribute, indices)

	split_value = randoms[random_index[0]]*(min_max_s.max_ - min_max_s.min_) + min_max_s.min_
	random_index[0] += 1
	
	"""
	Splitter: assign attribute and threshold
	"""
	splitter.attribute = attribute
	splitter.threshold = split_value

	"""
	Partition data with splitter
	"""
	partition(matrix, start, end, indices, splitter)

	free(Kurtosis_Values_log)

	return

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
cdef void assign_scores(DTYPE_t[:] scores, int *indices, int start, int end, double score):
	cdef int i
	
	for i from start <= i < end:
		scores[indices[i]] = -log(score)
	return

cdef void assign_depths(DTYPE_t[:] scores, int *indices, int start, int end, double score):
	cdef int i
	
	for i from start <= i < end:
		scores[indices[i]] = score
	return

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
cdef (double*, int) hash_matrix(DTYPE_t[:,::1] matrix, int n, int d):
	"""
	Hash data to obtain number of distinct values
	"""
	cdef double *hashed_ = <double *> malloc(n * sizeof(double))
	cdef DTYPE_t[:] randoms = np.random.uniform(size=d)
	cdef int i, j
	cdef set s = set()

	for i in range(n):
		hashed_[i] = 0
		for j in range(d):
			hashed_[i] += matrix[i][j]%randoms[j]
		
		s.add(hashed_[i])

	return (hashed_, len(s))

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
cdef int get_uniques(double *hashed_, int *indices, int start, int end):
	"""
	Get number of uniques
	"""
	cdef int i 
	cdef set s = set()

	for i from start <= i < end:
		s.add(hashed_[indices[i]])

	return len(s)


cdef class RHF():
	"""
	Random Histogram Forest
	"""
	cdef int n, d, global_uniques, num_trees, max_height, i, j, n_nodes, seed_state
	cdef DTYPE_t[:,::1] matrix
	cdef double *hashed_
	cdef int[:] seeds
	cdef double[:] scores, depths
	cdef DTYPE_t[:, :]  scores_all
	cdef RandomHistogramTree rht
	cdef bint check_duplicates, decremental, use_kurtosis
	# cdef double[:] indices_end

	def __init__(self, int num_trees = 100, int max_height = 5, check_duplicates = False, decremental = False, use_kurtosis=True, int seed_state = 100007):
		self.num_trees = num_trees
		self.max_height = max_height
		self.seed_state = seed_state
		if self.seed_state > 0:
			np.random.seed(self.seed_state)
		self.n_nodes = 2**(self.max_height+1)-1
		self.check_duplicates = check_duplicates
		self.decremental = decremental
		self.use_kurtosis = use_kurtosis
		# print('N Nodes: {}'.format(self.n_nodes))

	def __dealloc__(self):
		free(self.hashed_)

	def fit(self, matrix):
		self.fit_(np.ascontiguousarray(matrix, dtype=np.float64))

	def fit_(self, DTYPE_t[:,::1] matrix):

		# if matrix.dtype != DTYPE:
		# 	matrix = np.ascontiguousarray(matrix, dtype=DTYPE)


		self.matrix = matrix
		self.n = self.matrix.shape[0]
		self.d = self.matrix.shape[1]
		self.scores = np.zeros(self.n)
		self.depths = np.zeros(self.n)
		self.scores_all = np.zeros((self.num_trees, self.n))

		seeds = np.random.randint(np.iinfo(np.int32).max, size=self.num_trees)

		if self.check_duplicates:
			"""
			Check for duplicates
			Hash data
			"""
			self.hashed_, self.global_uniques = hash_matrix(self.matrix, self.n, self.d)
		else:
			"""
			Do not check duplicates
			"""
			self.global_uniques = self.n
		
		if self.decremental:
			"""
			Build RHT in decremental kurtosis mode
			"""
			for i in range(self.num_trees):
				rht = RandomHistogramTree(self.matrix, self.max_height, 0, seeds[i], self.global_uniques, self.n_nodes, self.check_duplicates)
				rht.build_decremental(self.hashed_)
				for j in range(self.n):
					self.scores[j] += rht.scores[j]
					self.depths[j] += rht.depths[j]
				self.scores_all[i] = rht.scores
		else:
			"""
			Build RHT
			"""
			for i in range(self.num_trees):
				rht = RandomHistogramTree(self.matrix, self.max_height, 0, seeds[i], self.global_uniques, self.n_nodes, self.check_duplicates, self.use_kurtosis)
				rht.build(self.hashed_)
				for j in range(self.n):
					self.scores[j] += rht.scores[j]
					self.depths[j] += rht.depths[j]
				self.scores_all[i] = rht.scores
		# with open('indices.json', 'w', encoding='utf-8') as f:
		# 	json.dump(rht.show_indices(), f, ensure_ascii=False, indent=4)

	def __dealloc__(self):
		free(self.hashed_)

	def get_global_uniques(self):
		return self.global_uniques

	def get_scores(self):
		return np.asarray(self.scores), np.asarray(self.scores_all)

	def get_depths(self):
		return np.asarray(self.depths)

	

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
cdef class RandomHistogramTree():
	"""
	Random Histogram Tree class
	"""
	cdef int n, d, global_uniques, n_nodes, nodes_count
	cdef DTYPE_t[:,::1] matrix
	cdef DTYPE_t[:] randoms
	cdef DTYPE_t[:] scores, depths
	cdef int *indices
	cdef double *hashed_
	cdef int i, random_index, max_height
	cdef bint debug, check_duplicates, use_kurtosis
	cdef int seed

	def __cinit__(self, DTYPE_t[:,::1] matrix, int max_height, bint debug, int seed, int global_uniques, int n_nodes, bint check_duplicates, bint use_kurtosis):
		self.matrix = matrix
		self.n = matrix.shape[0]
		self.d = matrix.shape[1]
		self.max_height = max_height
		self.scores = np.zeros(self.n, dtype=DTYPE)
		self.depths = np.zeros(self.n, dtype=DTYPE)
		self.seed = seed
		self.check_duplicates = check_duplicates
		self.use_kurtosis = use_kurtosis

		self.n_nodes = n_nodes
		self.nodes_count = 0

		"""
		Generate 2*n_nodes random numbers to use during split process
		Random_index is the index in the random array. Every time a random number is used, the index is increased
		"""
		self.random_index = 0
		np.random.seed(self.seed)
		self.randoms = np.random.uniform(size=2*self.n_nodes).astype(DTYPE)

		"""
		Indices of instances
		"""
		self.indices = initializeArangeArray(self.n)
		self.global_uniques = global_uniques

	def __dealloc__(self):
		free(self.indices)

	# @cython.boundscheck(False)
	# @cython.wraparound(False)
	# @cython.cdivision(True)
	cdef build(self, double *hashed_):

		self.hashed_ = hashed_
		"""
		Build function
		"""
		cdef Stack stack = Stack()
		cdef NodeRecord nodeRecord
		cdef Splitter splitter
		cdef int i
		cdef double node_uniques

		"""
		Push root in stack
		nodeRecord: start, end, depth, is_leaf
		"""
		code_result = stack.push(0, self.n, 0, 0)

		while not stack.is_empty():
			stack.pop(&nodeRecord)

			if nodeRecord.depth >= self.max_height:
				"""
				Leaf
				"""
				if self.check_duplicates:
					"""
					Assign score as number of distinct elements in leaf
					"""
					node_uniques = get_uniques(self.hashed_, self.indices, nodeRecord.start, nodeRecord.end)
					assign_scores(self.scores, self.indices, nodeRecord.start, nodeRecord.end, node_uniques/self.global_uniques)
					assign_depths(self.depths, self.indices, nodeRecord.start, nodeRecord.end, nodeRecord.depth)
				else:
					"""
					Assign score as number of elements in leaf
					"""
					assign_scores(self.scores, self.indices, nodeRecord.start, nodeRecord.end, (nodeRecord.end-nodeRecord.start)/self.global_uniques)
					assign_depths(self.depths, self.indices, nodeRecord.start, nodeRecord.end, nodeRecord.depth)

			elif nodeRecord.end - nodeRecord.start == 1:
				"""
				Leaf
				"""
				node_uniques = 1
				assign_scores(self.scores, self.indices, nodeRecord.start, nodeRecord.end, node_uniques/self.global_uniques)
				assign_depths(self.depths, self.indices, nodeRecord.start, nodeRecord.end, nodeRecord.depth)

			else:
				if self.use_kurtosis:
					kurtosis_split(self.matrix, self.d, self.indices, nodeRecord.start, nodeRecord.end, &splitter, self.randoms, &self.random_index)
				else:
					random_split(self.matrix, self.d, self.indices, nodeRecord.start, nodeRecord.end, &splitter, self.randoms, &self.random_index)

				if splitter.attribute == -1:
					"""
					Leaf: Impossible to SPLIT
					"""
					if self.check_duplicates:
						"""
						Assign score as number of distinct elements in leaf
						"""
						node_uniques = get_uniques(self.hashed_, self.indices, nodeRecord.start, nodeRecord.end)
						assign_scores(self.scores, self.indices, nodeRecord.start, nodeRecord.end, node_uniques/self.global_uniques)
						assign_depths(self.depths, self.indices, nodeRecord.start, nodeRecord.end, nodeRecord.depth)
					else:
						"""
						Assign score as number of elements in leaf
						"""
						assign_scores(self.scores, self.indices, nodeRecord.start, nodeRecord.end, (nodeRecord.end-nodeRecord.start)/self.global_uniques)
						assign_depths(self.depths, self.indices, nodeRecord.start, nodeRecord.end, nodeRecord.depth)
				else:

					"""
					Left Branch
					"""
					code_result = stack.push(nodeRecord.start, splitter.position, nodeRecord.depth+1, 0)

					"""
					Right Branch
					"""
					code_result = stack.push(splitter.position, nodeRecord.end, nodeRecord.depth+1, 0)

	# @cython.boundscheck(False)
	# @cython.wraparound(False)
	# @cython.cdivision(True)
	cdef build_decremental(self, double *hashed_):

		self.hashed_ = hashed_
		"""
		Build function
		"""

		cdef StackKurt stack = StackKurt()
		cdef NodeRecordKurt nodeRecord
		cdef Splitter splitter
		cdef int i, ii_print
		cdef double size
		cdef double node_uniques

		"""
		Push root in stack
		nodeRecord: start, end, depth, is_leaf
		"""
		cdef kurtosis_struct **nodes_kstruct = init_kstruct_array(self.n_nodes, self.d)
		computeKurtosis(self.matrix, self.d, 0, self.n, self.indices, nodes_kstruct[0])

		code_result = stack.push(0, self.n, 0, 0, self.nodes_count)


		while not stack.is_empty():
			stack.pop(&nodeRecord)

			if debug:
				print('')
				print('Extracting from Stack: ID:{}-Start:{}-End:{}-Depth:{}'.format(nodeRecord.nodeID, nodeRecord.start, nodeRecord.end, nodeRecord.depth))
			# print('Kurtosis Struct: n: {} - k: {} | n: {} - k: {} | n: {} - k: {}'.format(nodes_kstruct[nodeRecord.nodeID][0].n, round(nodes_kstruct[nodeRecord.nodeID][0].kvalue,6),
			# 	nodes_kstruct[nodeRecord.nodeID][1].n, round(nodes_kstruct[nodeRecord.nodeID][1].kvalue,6),
			# 	nodes_kstruct[nodeRecord.nodeID][2].n, round(nodes_kstruct[nodeRecord.nodeID][2].kvalue,6)))

			if debug:
				for ii_print in range(3):
					print('Kurt Struct! n:{} | M2: {} | M4: {} | kv: {}'.format(
						nodes_kstruct[nodeRecord.nodeID][ii_print].n,
						round(nodes_kstruct[nodeRecord.nodeID][ii_print].M2, 6),
						round(nodes_kstruct[nodeRecord.nodeID][ii_print].M4, 6),
						round(nodes_kstruct[nodeRecord.nodeID][ii_print].kvalue, 6)))


			if nodeRecord.depth >= self.max_height:
				"""
				Leaf
				"""
				if self.check_duplicates:
					"""
					Assign score as number of distinct elements in leaf
					"""
					node_uniques = get_uniques(self.hashed_, self.indices, nodeRecord.start, nodeRecord.end)
					assign_scores(self.scores, self.indices, nodeRecord.start, nodeRecord.end, node_uniques/self.global_uniques)
				else:
					"""
					Assign score as number of elements in leaf
					"""
					assign_scores(self.scores, self.indices, nodeRecord.start, nodeRecord.end, (nodeRecord.end-nodeRecord.start)/self.global_uniques)
					if debug:
						print('Max depth! Return LEAF!')

			elif nodeRecord.end - nodeRecord.start == 1:
				"""
				Leaf
				"""
				node_uniques = 1
				assign_scores(self.scores, self.indices, nodeRecord.start, nodeRecord.end, node_uniques/self.global_uniques)
				if debug:
					print('One Record! Return LEAF!')

			else:

				decremental_kurtosis_split(self.matrix, self.d, self.indices, nodeRecord.start, nodeRecord.end, &splitter, self.randoms, &self.random_index, nodes_kstruct, nodeRecord.nodeID, self.nodes_count)

				if splitter.attribute == -1:
					"""
					Leaf: Impossible to SPLIT anymore => LEAF
					"""
					if self.check_duplicates:
						"""
						Assign score as number of distinct elements in leaf
						"""
						node_uniques = get_uniques(self.hashed_, self.indices, nodeRecord.start, nodeRecord.end)
						assign_scores(self.scores, self.indices, nodeRecord.start, nodeRecord.end, node_uniques/self.global_uniques)
					else:
						"""
						Assign score as number of elements in leaf
						"""
						assign_scores(self.scores, self.indices, nodeRecord.start, nodeRecord.end, (nodeRecord.end-nodeRecord.start)/self.global_uniques)
				else:
					self.nodes_count+= 1
					code_result = stack.push(nodeRecord.start, splitter.position, nodeRecord.depth+1, 0, self.nodes_count)

					self.nodes_count+= 1
					code_result = stack.push(splitter.position, nodeRecord.end, nodeRecord.depth+1, 0, self.nodes_count)

		for i in range(self.n_nodes):
			free(nodes_kstruct[i])
		free(nodes_kstruct)

	def show_indices(self):
		results = []
		for x in range(self.n):
			results.append(self.indices[x])
		
		return results

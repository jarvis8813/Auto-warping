from util import *
from skimage import morphology,draw
import matplotlib.pyplot as plt
import copy


class Skeleton:

	def __init__(self, eros_pic):
		self.img = np.zeros_like(eros_pic)
		self.list = []

	def add_point(self, point):
		self.list.append([point[1],point[0]])
		self.img[point[0],point[1]] = 1

def find_green_pnt(img):
	green_dist = np.sum(np.square(img - np.array([[0.,255.,0.]])),axis=2)
	idx_x, idx_y = np.where(green_dist==0)
	idx_x = np.asarray(idx_x)
	idx_y = np.asarray(idx_y)
	central_x = np.mean(idx_x)
	central_y = np.mean(idx_y)

	return np.array([central_x,central_y])

def find_start_pnt(sket):
	coord_x, coord_y = np.where(sket == 1)
	start_idx = np.argmin(coord_x)
	now_pnt = np.array([coord_x[start_idx],coord_y[start_idx]])
	last_pnt = now_pnt
	connec, nxt_pnt = find_nxt_pnt(sket, now_pnt, last_pnt)
	direc_list = []
	direc_list = direc_list_update(direc_list, now_pnt)
	# return now_pnt
	if connec!=1:
		direc_vec = cal_direc(direc_list)
		nxt_pnt = selec_nxt_pnt(direc_vec, now_pnt, nxt_pnt)
		last_pnt = now_pnt
		now_pnt = nxt_pnt
		direc_list = direc_list_update(direc_list, now_pnt)
		connec, nxt_pnt = find_nxt_pnt(sket, now_pnt, last_pnt)
		
		while connec!=0:
			if connec == 1:
				last_pnt = now_pnt
				now_pnt = nxt_pnt[0]
				direc_list = direc_list_update(direc_list, now_pnt)
				connec, nxt_pnt = find_nxt_pnt(sket, now_pnt, last_pnt)
			elif connec >= 2:
				direc_vec = cal_direc(direc_list)
				nxt_pnt = selec_nxt_pnt(direc_vec, now_pnt, nxt_pnt)
				last_pnt = now_pnt
				now_pnt = nxt_pnt
				direc_list = direc_list_update(direc_list, now_pnt)
				connec, nxt_pnt = find_nxt_pnt(sket, now_pnt, last_pnt)
		return now_pnt
	else:
		return now_pnt


def find_nxt_pnt(sket, now_pnt, last_pnt):
	# print('now_pnt',now_pnt)
	center_x = now_pnt[0]
	center_y = now_pnt[1]
	neighbor = copy.deepcopy(sket[center_x-1:center_x+2,center_y-1:center_y+2])
	# set last_point and now_point to zero, the rest will be new branch
	
	neighbor[1,1] = 0
	neighbor[last_pnt[0]-center_x+1,last_pnt[1]-center_y+1] = 0
	idx_x, idx_y = np.where(neighbor == 1)
	idx_x = center_x + idx_x -1
	idx_y = center_y + idx_y -1
	nxt_pnt = [np.asarray([idx_x[i],idx_y[i]]) for i in xrange(len(idx_x))]
	connec = len(nxt_pnt)

	return connec, nxt_pnt

def selec_nxt_pnt(direc_vec, now_pnt, nxt_pnt):
	# calculate the vector, and the intersection angle betwwen two vectors
	# choose the minimum angle
	# print("direc_vec",direc_vec)
	product_list = []
	unit_direc_vec = direc_vec/np.sqrt(np.sum(np.square(direc_vec)))
	# print('unit_direc_vec',unit_direc_vec)
	for i in xrange(len(nxt_pnt)):
		pnt_i = nxt_pnt[i]
		branch_vec = pnt_i - now_pnt
		unit_branch_vec = branch_vec/np.sqrt(np.sum(np.square(branch_vec)))
		# print('unit_branch_vec',unit_branch_vec)
		product_list.append(np.sum(unit_direc_vec*unit_branch_vec))
	product_list = np.array(product_list)
	# the larger cross product is, the smaller angle is
	# print('next points',nxt_pnt)
	# print('cross product',product_list)
	min_prod = np.argmax(product_list)
	choose_pnt = nxt_pnt[min_prod]

	return choose_pnt


def direc_list_update(direc_list, pnt):
	max_len = 15
	if len(direc_list) < max_len:
		direc_list.append(pnt)
	elif len(direc_list) == max_len:
		direc_list.append(pnt)
		del direc_list[0]
	elif len(direc_list) >max_len:
		raise ValueError("the number of points in direc_list has exceeded" )
	return direc_list


def cal_direc(direc_list):
	# print('direc_list',direc_list)
	N = len(direc_list)
	direc_vec = np.array([0.,0.])
	if N == 1:
		return direc_vec
	else:		
		for i in xrange(N-1):
			delta_i = direc_list[i+1] - direc_list[i]
			direc_vec += delta_i
		direc_vec = direc_vec/(N-1)
		return direc_vec


def find_main_sket(sket):
	# need a more specific way to definite start point!!!!!!!!

	start_pnt = find_start_pnt(sket)
	
	# use class Skeleton
	main_sket = Skeleton(sket)
	main_sket.add_point(start_pnt)
	# use a list to store 5 last point to calculate a direction
	direc_list = []
	direc_list = direc_list_update(direc_list, start_pnt)

	last_pnt = start_pnt
	now_pnt = start_pnt
	connec, nxt_pnt = find_nxt_pnt(sket, now_pnt, last_pnt)
	
	while connec != 0:
		if connec == 1:
			last_pnt = now_pnt
			now_pnt = nxt_pnt[0]
			direc_list = direc_list_update(direc_list, now_pnt)
			connec, nxt_pnt = find_nxt_pnt(sket, now_pnt, last_pnt)
			main_sket.add_point(now_pnt)
		elif connec >= 2:
			direc_vec = cal_direc(direc_list)
			nxt_pnt = selec_nxt_pnt(direc_vec, now_pnt, nxt_pnt)
			last_pnt = now_pnt
			now_pnt = nxt_pnt
			direc_list = direc_list_update(direc_list, now_pnt)
			connec, nxt_pnt = find_nxt_pnt(sket, now_pnt, last_pnt)
			main_sket.add_point(now_pnt)

	main_sket.list = np.asarray(main_sket.list)
	return main_sket


def get_main_sket(img):
	green_p = find_green_pnt(img)
	print("green point!",green_p)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	_,bi = cv2.threshold(gray,240,1,cv2.THRESH_BINARY_INV)


	skeleton = morphology.skeletonize(bi)

	## find the main skeleton
	# use the top point as start

	main_sket = find_main_sket(skeleton)


	# fig, (ax1, ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

	# ax1.imshow(img, cmap=plt.cm.gray)
	# ax1.axis('off')
	# ax1.set_title('original', fontsize=20)

	# ax2.imshow(skeleton, cmap=plt.cm.gray)
	# ax2.axis('off')
	# ax2.set_title('skeleton', fontsize=20)

	# ax3.imshow(main_sket.img, cmap=plt.cm.gray)
	# ax3.axis('off')
	# ax3.set_title('main', fontsize=20)

	# fig.tight_layout()
	# plt.show()

	return main_sket

def get_sket(img):
	green_p = find_green_pnt(img)
	print("green point!",green_p)

	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# print("here",gray[green_p])
	_,bi = cv2.threshold(gray,240,1,cv2.THRESH_BINARY_INV)


	skeleton = morphology.skeletonize(bi)

	## find the main skeleton
	# use the top point as start

	main_sket = skeleton


	fig, (ax1, ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

	ax1.imshow(img, cmap=plt.cm.gray)
	ax1.axis('off')
	ax1.set_title('original', fontsize=20)

	ax2.imshow(skeleton, cmap=plt.cm.gray)
	ax2.axis('off')
	ax2.set_title('skeleton', fontsize=20)

	# ax3.imshow(main_sket.img, cmap=plt.cm.gray)
	# ax3.axis('off')
	# ax3.set_title('main', fontsize=20)

	fig.tight_layout()
	plt.show()

	return main_sket

def get_angle(vec):
	y = -vec[1]
	x = vec[0]
	angle = np.arctan2(y,x)
	return angle

def find_pivot(src_sket, control_p):
	sket_vec = src_sket.list
	M = len(sket_vec)
	N = len(control_p)
	# use broadcast to calculate the distance 
	pivot_parameter = {}
	sket_vec = np.expand_dims(sket_vec,axis=0)
	control_p = np.expand_dims(control_p, axis=1)
	dist_matrix = np.sqrt(np.sum(np.square(sket_vec-control_p),axis=2))		# int????float???
	print('N',dist_matrix.shape)

	dist = np.min(dist_matrix,axis=1)
	pivot_p = np.argmin(dist_matrix, axis=1)

	control_p = np.squeeze(control_p)
	sket_vec = np.squeeze(sket_vec)
	angle = []
	
	print("sket_vec",sket_vec.shape)
	for i in xrange(len(pivot_p)):
		stem_p = sket_vec[pivot_p[i]]
		if pivot_p[i] == 0:
			stem_vec = sket_vec[0] - sket_vec[2]
		elif pivot_p[i] == (len(sket_vec)-1):
			stem_vec = sket_vec[-3] - sket_vec[-1]
			# print("pivot_p[i] last",pivot_p[i])
		else:
			stem_vec = sket_vec[pivot_p[i]-1] - sket_vec[pivot_p[i]+1]
		branch_vec = control_p[i] - stem_p
		angle_i = get_angle(branch_vec) - get_angle(stem_vec) 
		angle.append(angle_i)

	angle = np.asarray(angle)
	pivot_parameter['pivot_p'] = pivot_p
	pivot_parameter['dist'] = dist
	pivot_parameter['angle'] = angle
	
	return pivot_parameter

def cal_length(sket):
	length = 0
	for i in xrange(len(sket)-1):
		unit_length = np.sqrt(np.sum(np.square(sket[i+1]- sket[i])))
		length += unit_length

	return length


def sket_length_ratio(src_sket,trg_sket):
	
	src_length = np.float(len(src_sket.list))
	trg_length = np.float(len(trg_sket.list))

	ratio = trg_length / src_length
	print("ratio",ratio)
	return ratio


def normalize_pivot(pivot_parameter, ratio):
	pivot_p = pivot_parameter['pivot_p']
	pivot_p = np.round(pivot_p * ratio)
	pivot_p = pivot_p.astype(int)
	pivot_parameter['pivot_p'] = pivot_p
	pivot_parameter['dist'] = pivot_parameter['dist'] * ratio

	return pivot_parameter

def new_control_point(pivot_parameter, sket):
	pivot_p = pivot_parameter['pivot_p']
	dist = pivot_parameter['dist']
	angle = pivot_parameter['angle']
	
	sket_vec = sket.list
	stem_p = sket_vec[pivot_p]
	new_p = np.zeros_like(stem_p)
	base_angle = np.zeros_like(angle)

	for i in xrange(len(pivot_p)):
		if pivot_p[i] == 0:
			stem_vec = sket_vec[0] - sket_vec[2]
		elif pivot_p[i] == (len(sket_vec)-1):
			stem_vec = sket_vec[-3] - sket_vec[-1]
		else:
			stem_vec = sket_vec[pivot_p[i]-1] - sket_vec[pivot_p[i]+1]
		base_angle[i] = get_angle(stem_vec)

	theta = base_angle + angle
	new_p[:,0] = stem_p[:,0] + dist * np.cos(theta)
	new_p[:,1] = stem_p[:,1] - dist * np.sin(theta)

	return new_p

if __name__ == '__main__':
	img = cv2.imread("./test_images/S10.jpg")
	get_sket(img)
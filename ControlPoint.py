from util import *
from delaunay2D import Delaunay2D
# Useful links:
# http://stackoverflow.com/questions/3862225/implementing-a-harris-corner-detector
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html

def harris(src, blockSize, ksize, k):
  def harris_score(M):
    return np.linalg.det(M) - k * (np.trace(M) ** 2)

  return generate_gradient_matrix(src, blockSize, ksize, k, harris_score)

def max_suppression(corners):
	W,H = corners.shape
	result = np.zeros_like(corners)
	for i in xrange(W):
		for j in xrange(H):
			if corners[i,j] == 0:
				continue
			if corners[i,j] != 0:
				if corners[i,j] == np.max(corners[i-2:i+2,j-2:j+2]):
					result[i,j] = 1
	return result

def find_corners(gray):
	t_opencv = time()
	print "TRY: Running OpenCV Harris corner detector"
	corners = cv2.cornerHarris(gray, 2, 3, 0.04)
	print "SUCCESS (%.3f secs): Running OpenCV Harris corner detector" % (time() - t_opencv)
	corners[corners < 0.1 * corners.max()] = 0
	#corners simplify
	result = max_suppression(corners) 
	return result

def find_contours(gray):
	_,bi = cv2.threshold(gray,250,255,cv2.THRESH_BINARY_INV)
	vals = bi
	contours=cv2.findContours(vals, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, (0,0))
	point_vec = contours[0][-1]
	N = point_vec.shape[0]
	point_list = []
	ptr = i = 0
	point_list.append(point_vec[0])
	def dist(x,y):
		return np.sqrt(np.sum(np.square(x-y)))
	while i<N:
		if dist(point_vec[ptr],point_vec[i]) > 7:
			point_list.append(point_vec[i])
			ptr = i
		i +=1
	point_list = np.array(point_list)
	point_list = np.squeeze(point_list)
	return point_list

# Code in this method modified from:
# http://docs.opencv.org/trunk/doc/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
def write_image_with_control_points_and_show(name, image, control_points):
  # result is dilated for marking the corners, not important
  # result = cv2.dilate(corners, None)


  # Threshold for an optimal value, it may vary depending on the image.
  image[control_points != 0] = [0, 0, 255]

  cv2.imwrite("%s.png" % name, image)
  # cv2.imshow(name, image)


def control_point(img):

	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	# to find corner points and the extreme points
	corners = find_corners(gray)

	corner_vec = img_to_vec(corners)
	corner_min = np.argmin(corner_vec,axis=0)
	corner_max = np.argmax(corner_vec,axis=0)
	left = corner_vec[corner_min[0]]
	up = corner_vec[corner_min[1]]
	right = corner_vec[corner_max[0]]
	down = corner_vec[corner_max[1]]
	corner_vec = np.append(corner_vec,[left,up,right,down],axis=0)
	print("corner points num",corner_vec.shape[0])
	# find the contour and extract points evenly
	contour_vec = find_contours(gray)
	print("contour points num",contour_vec.shape[0])
	# spit out some rebundant points from contour vec
	control_p_vec = contour_vec
	for corner in list(corner_vec):
		N = len(control_p_vec)
		dist = np.sqrt(np.sum(np.square(control_p_vec-corner),axis=1))
		dist_min = np.argmin(dist)
		if dist[(dist_min-1)%N] < dist[(dist_min+1)%N]:
			control_p_vec = np.insert(control_p_vec,dist_min, corner, axis=0)
			if dist[dist_min]<7:
				control_p_vec = np.delete(control_p_vec,(dist_min+1)%N, axis=0)
			if dist[(dist_min-1)%N] < 7:
				control_p_vec = np.delete(control_p_vec,(dist_min-1)%N, axis=0)
		elif dist[(dist_min-1)%N] >= dist[(dist_min+1)%N]:
			control_p_vec = np.insert(control_p_vec,(dist_min+1)%N, corner, axis=0)
			if dist[dist_min]<7:
				control_p_vec = np.delete(control_p_vec,dist_min, axis=0)
			if dist[(dist_min+1)%N] < 7:
				control_p_vec = np.delete(control_p_vec,(dist_min+2)%N, axis=0)	

	print("control points num",len(control_p_vec))
	control_p_img = vec_to_img(contour_vec,img)
	
	# # if you want to display the control points
	# write_image_with_control_points_and_show("result", img, control_p_img)
	
	# if noly use contour vector
	control_p_vec = contour_vec
	return control_p_vec

if __name__ == '__main__':
	control_point()
	# # apply triangulation
	# # print("control_points:\n", control_p_vec)
	# print("BBox Min:", np.amin(control_p_vec, axis=0),
	#       "Bbox Max: ", np.amax(control_p_vec, axis=0))

	# """
	# Compute our Delaunay triangulation of control_points.
	# """
	# # It is recommended to build a frame taylored for our data
	# # dt = D.Delaunay2D() # Default frame
	# center = np.mean(control_p_vec, axis=0)
	# dt = Delaunay2D(center)

	# # Insert all control_points one by one
	# for s in control_p_vec:
	#     dt.addPoint(s)

	# # Dump number of DT triangles
	# print (len(dt.exportTriangles()), "Delaunay triangles")
	# print(dt.exportTriangles())
	# """
	# Demostration of how to plot the data.
	# """
	# import matplotlib.pyplot as plt
	# import matplotlib.tri
	# import matplotlib.collections

	# # Plot annotated Delaunay vertex (seeds)
	# for i, v in enumerate(control_p_vec):
	#     plt.annotate(i, xy=v)

	# # Create a plot with matplotlib.pyplot
	# fig, ax = plt.subplots()
	# ax.margins(0.1)
	# ax.set_aspect('equal')
	# plt.axis([-1, 256, -1, 256])

	# # Plot our Delaunay triangulation (plot in blue)
	# cx, cy = zip(*control_p_vec)
	# dt_tris = dt.exportTriangles()
	# ax.triplot(matplotlib.tri.Triangulation(cx, cy, dt_tris), 'bo--')

	# plt.show()
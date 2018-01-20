from util import *
import numpy as np
import sys
from ControlPoint import *
from skeleton import *
from delaunay2D import Delaunay2D
from transform import *
import matplotlib.pyplot as plt
import matplotlib.tri
import matplotlib.collections


def main():
	
	src_img, gray = read_image(srcimg_name)
	trg_img,_ = read_image(trgimg_name)
	control_p = control_point(src_img)  # shape[N,2]
	src_sket = get_main_sket(src_img)			# shape[image_shape]
	trg_sket = get_main_sket(trg_img)
	pivot_parameter = find_pivot(src_sket, control_p)

 	ratio = sket_length_ratio(src_sket,trg_sket)
	norm_parameter = normalize_pivot(pivot_parameter, ratio)


	n_control_p = new_control_point(norm_parameter, trg_sket)


	# apply triangulation
	# print("controlpivot_points:\n", control_p)
	print("BBox Min:", np.amin(control_p, axis=0),
	      "Bbox Max: ", np.amax(control_p, axis=0))

	"""
	Compute our Delaunay triangulation of control_points.
	"""
	# It is recommended to build a frame taylored for our data
	# dt = D.Delaunay2D() # Default frame
	center = np.mean(control_p, axis=0)
	dt = Delaunay2D(center)

	# Insert all control_points one by one
	for s in control_p:
	    dt.addPoint(s)

	# Dump number of DT triangles
	print (len(dt.exportTriangles()), "Delaunay triangles")
	print(dt.exportTriangles())

	triangle_idx = dt.exportTriangles()
	
	out_img = apply_transform(src_img,control_p,n_control_p, triangle_idx)



	fig, (ax1, ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

	ax1.imshow(src_img)
	ax1.axis('off')
	ax1.set_title('original', fontsize=20)

	ax2.imshow(trg_img)
	ax2.axis('off')
	ax2.set_title('skeleton', fontsize=20)

	ax3.imshow(out_img)
	ax3.axis('off')
	ax3.set_title('main', fontsize=20)

	fig.tight_layout()
	plt.show()

	output = np.concatenate([src_img,trg_img,out_img],axis=1)
	cv2.imwrite("output.jpg",output)

	"""
	Demostration of how to plot the data.
	"""
	# # # Plot annotated Delaunay vertex (seeds)
	# for i, v in enumerate(control_p):
	#     plt.annotate(i, xy=v)

	# # Create a plot with matplotlib.pyplot
	# fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)
	# ax1.margins(0.1)
	# ax1.set_aspect('equal')
	# plt.axis([-1, 256, -1, 256])

	# # Plot our Delaunay triangulation (plot in blue)
	# cx, cy = zip(*control_p)
	# dt_tris = dt.exportTriangles()
	# ax1.triplot(matplotlib.tri.Triangulation(cx, cy, dt_tris), 'bo--')

	# ax2.scatter(n_control_p[:,0],n_control_p[:,1])
	# plt.show()




if __name__ == '__main__':
	if len(sys.argv) >1 :
		img_name = sys.argv[1]
	else:
		srcimg_name = './test_images/P1.jpg'
		trgimg_name = './test_images/S12.jpg'
	main()


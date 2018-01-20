import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import sys

def applyAffineTransform(src, srcTri, dstTri, size) :    
    
    M = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    dst = cv2.warpAffine( src, M , (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 ) 

    return dst

def warpTriangle(img1, img2, t1, t2) :   
	# img1:input image. img2:output image, ti:input triangle coordinate, t2:output triangle coordinate
    # Find bounding rectangle for each triangle
    channel = img1.shape[2]
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in range(3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], channel), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

    if len(img2Rect.shape) == 2:
    	img2Rect = np.expand_dims(img2Rect, axis=2)

    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    # img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )

    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect

def apply_transform(src_img,control_p,n_control_p, triangle_idx):
	
	gray = cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)
	_,bi = cv2.threshold(gray,250,1,cv2.THRESH_BINARY_INV)
	bi = np.expand_dims(bi, axis=2)
	# bi = np.tile(bi,[1,1,3])
	src_img = src_img*bi  
	out_img = np.zeros_like(src_img,np.float32)
	overlap = np.zeros_like(bi)

	for i in xrange(len(triangle_idx)):
		tin = []
		tout = []
		for j in range(3) :
		    pIn = control_p[triangle_idx[i][j]]
		    pOut= n_control_p[triangle_idx[i][j]]

		    tin.append(pIn)
		    tout.append(pOut)
		    
		warpTriangle(src_img, out_img, tin, tout)	
		warpTriangle(bi, overlap, tin, tout)
		# fig, (ax1, ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
		# ax1.imshow(src_img)
		# ax1.axis('off')
		# ax1.set_title('1', fontsize=20)
		# ax2.imshow(out_img)
		# ax2.axis('off')
		# ax2.set_title('2', fontsize=20)
		# ax3.imshow(overlap)		
		# ax3.axis('off')
		# ax3.set_title('3', fontsize=20)
		# plt.show()
	overlap = np.where(overlap==0, np.ones_like(overlap),overlap)
	# overlap = np.tile(overlap,[1,1,3])
	print("overlap",overlap.max(),"shape",overlap.shape)
	out_img = np.where(out_img==0, 255*np.ones_like(out_img),out_img)
	
	output = out_img/overlap
	overlap = np.squeeze(overlap)

	fig, (ax1, ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
	ax1.imshow(src_img)
	ax1.axis('off')
	ax1.set_title('1', fontsize=20)
	ax2.imshow(output)
	ax2.axis('off')
	ax2.set_title('2', fontsize=20)
	im = ax3.imshow(overlap,cmap=plt.cm.gray)	
	plt.colorbar(im)	
	ax3.axis('off')
	ax3.set_title('3', fontsize=20)
	plt.show()


	return output
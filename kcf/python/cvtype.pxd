cdef extern from "opencv2/core/cvdef.h":
	cdef int CV_8U
	cdef int CV_8S
	cdef int CV_16U
	cdef int CV_16S
	cdef int CV_32S
	cdef int CV_32F
	cdef int CV_64F
	cdef int CV_MAKETYPE(int, int)
	cdef int CV_MAT_DEPTH(int)
	cdef int CV_MAT_CN(int)

cdef extern from "opencv2/core/mat.hpp" namespace "cv":
	cdef cppclass Mat:
		Mat()
		Mat(int, int, int)
		Mat(int, int, int, void*, int)
		int type()
		void* data
		int cols
		int rows
		
cdef extern from "opencv2/core/types.hpp" namespace "cv":
	cdef cppclass Point:
		Point()
		Point(int, int)
		int x, y
	cdef cppclass Size:
		Size()
		Size(int, int)
		int width, height
	cdef cppclass Rect:
		Rect()
		Rect(int, int, int, int)
		int x, y, width, height

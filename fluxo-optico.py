import cv2
import numpy as np

if __name__ == '__main__':

	cam = cv2.VideoCapture('movement-detection.mov')
	font = cv2.FONT_HERSHEY_SIMPLEX

	def draw(flow):
		(h, w) = flow.shape[:2]
		mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
		hsv = np.zeros([h, w, 3], np.uint8)
		hsv[..., 0] = ang * (180 / np.pi / 2)
		hsv[..., 1] = 0xFF
		hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

		flowImg = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

		return flowImg

	threshold = 5
	scale = .6
	grava = True

	ret, prev = cam.read()

	prev = cv2.resize(prev, None, fx = scale, fy = scale)

	prevGray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter('movement.mp4', fourcc, 20.0, (prev.shape[1], prev.shape[0] * 2))

	while True:

		ret, img = cam.read()

		if not ret:
			exit()

		img = cv2.resize(img, None, fx = scale, fy = scale)

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		flow = cv2.calcOpticalFlowFarneback(prevGray, gray, None, 0.7, 30, threshold, 1, 1, 1.02, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

		prevGray = gray

		flowImg = draw(flow)

		redLower = np.array([10, 10, 60], dtype = np.uint8)
		redUpper = np.array([0, 0, 2], dtype = np.uint8)

		greenLower = np.array([30, 60, 0], dtype = np.uint8)
		greenUpper = np.array([2, 2, 0], dtype = np.uint8)

		violetLower = np.array([30, 0, 60], dtype = np.uint8)
		violetUpper = np.array([2,  0, 2], dtype = np.uint8)

		orangeLower = np.array([0, 50, 40], dtype = np.uint8)
		orangeUpper = np.array([0,  2, 2], dtype = np.uint8)

		maskRed = cv2.inRange(flowImg, redUpper, redLower)
		maskGreen = cv2.inRange(flowImg, greenUpper, greenLower)
		maskViolet = cv2.inRange(flowImg, violetUpper, violetLower)
		maskOrange = cv2.inRange(flowImg, orangeUpper, orangeLower)

		contoursRed, hierarchyRed = cv2.findContours(maskRed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contoursGreen, hierarchyGreen = cv2.findContours(maskGreen, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contoursViolet, hierarchyViolet = cv2.findContours(maskViolet, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contoursOrange, hierarchyViolet = cv2.findContours(maskOrange, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		for cnts in contoursRed:
			(x, y, w, h) = cv2.boundingRect(cnts)
			if w > 100 and h > 200:
				cv2.rectangle(img, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 255, 0), 2, cv2.LINE_AA)
				cv2.putText(flowImg, 'Right', (25, 25), font, .5, [255, 255, 255], 1, cv2.LINE_AA)

		for cnts in contoursGreen:
			(x, y, w, h) = cv2.boundingRect(cnts)
			if w > 100 and h > 200:
				cv2.rectangle(img, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 255, 0), 2, cv2.LINE_AA)
				cv2.putText(flowImg, 'Left', (25, 35), font, .5, [255, 255, 255], 1, cv2.LINE_AA)

		for cnts in contoursViolet:
			(x, y, w, h) = cv2.boundingRect(cnts)
			if w > 100 and h > 200:
				cv2.rectangle(img, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 255, 0), 2, cv2.LINE_AA)
				cv2.putText(flowImg, 'Up', (25, 45), font, .5, [255, 255, 255], 1, cv2.LINE_AA)

		for cnts in contoursOrange:
			(x, y, w, h) = cv2.boundingRect(cnts)
			if w > 100 and h > 200:
				cv2.rectangle(img, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 255, 0), 2, cv2.LINE_AA)
				cv2.putText(flowImg, 'Down', (25, 55), font, .5, [255, 255, 255], 1, cv2.LINE_AA)

		final = cv2.vconcat([img, flowImg])

		cv2.imshow('final', final)

		if grava == True:
			out.write(final)

		ch = cv2.waitKey(15)

		if ch == ord('q'):
			out.release()
			cam.release()
			cv2.destroyAllWindows()
			break

	out.release()
	cam.release()
	cv2.destroyAllWindows()

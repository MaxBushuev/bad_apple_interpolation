import time
from time import sleep
import io

import numpy as np
import cv2
from scipy.interpolate import splprep, splev
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_inter_curves(frame):
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    curves = []
    for contour in contours:
        contour = np.reshape(contour, (-1, 2))

        try:
            tck, u = splprep(contour.T, u=None, s=2., per=1)
            x, y = splev(u, tck, der=0)

            curves.append((x, y))
        except:
            continue
        
    return curves

def render_frame(figure, curves):
    plt.clf()
    if curves:
        for x, y in curves:
            plt.plot(x, y, 'b')

            plt.xlim(0, 480)
            plt.ylim(0, 360)

            plt.pause(0.001)

    io_buf = io.BytesIO()
    figure.savefig(io_buf, format='raw')
    io_buf.seek(0)

    frame = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                       newshape=(int(figure.bbox.bounds[3]), int(figure.bbox.bounds[2]), -1))
    io_buf.close()

    return frame
    

def main():
    cap = cv2.VideoCapture('bad_apple.mp4')
    assert cap.isOpened(), "Can't open the video"

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('bad_apple_interpolated.avi', fourcc, 30.0, (640, 480))

    matplotlib.use('Agg')
    fig = plt.figure()

    pbar = tqdm(total=6600)

    while True:
        ret, frame = cap.read()
        if not ret:
            print('The video is ready!')
            break

        frame = cv2.flip(frame, 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        curves = get_inter_curves(frame)
        rendered_frame = render_frame(fig, curves)
        rendered_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_RGBA2BGR)
        out.write(rendered_frame)
        pbar.update(1)

        cv2.waitKey(1)

    cap.release()
    out.release()
    

if __name__ == '__main__':
    main()

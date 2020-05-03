
from datetime import datetime


def dlib_rectangle_to_xywh(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def timestamp(name):
    return "%s_%s" \
        % (datetime.now().strftime('%Y-%m-%dT%H-%M-%S'), name)


if __name__ == '__main__':
    pass

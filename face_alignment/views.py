def get_view(bbox, landmark):
    views = ['straight', 'left', 'right', 'up', 'down']
    x, y, xw, yh = bbox[:4]
    x_nose, y_nose = landmark[2]

    diff_left = abs(x_nose - x)
    diff_right = abs(xw - x_nose)
    diff_up = abs(y_nose - y)
    diff_down = abs(yh - y_nose)
    t1 = 2.5
    # t2 = 1

    view = views[0]
    ratio = 1

    if diff_left > diff_right:
        ratio = diff_left / diff_right
        if (ratio > t1):
            view = views[1]
    elif diff_left < diff_right:
        ratio = diff_right / diff_left
        if (ratio > t1):
            view = views[2]
    
    if diff_up > diff_down:
        ratio = diff_up / diff_down
        if (ratio > 1.5):
            view = views[4]
    elif diff_up < diff_down:
        ratio = diff_down / diff_up
        if (ratio > 1):
            view = views[3]
        
    return f'{view} - {ratio:.2f}'



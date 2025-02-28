import numpy as np

def get_location(img, idx, th=0.7):
    channel_bool = (img[:,:,idx] > th) & (img[:,:, (idx+1)%3] < th) & (img[:,:, (idx+2)%3] < th)
    location = np.where(channel_bool == True)
    location = np.array(location)
    return location

def get_minH_maxF(image):
    minH = 0
    maxF = 9999
    minHmaxF = np.zeros([3, 2])

    r_location = get_location(image, 0)
    g_location = get_location(image, 1)
    b_location = get_location(image, 2)

    if r_location.shape[1] == 0:
        print('there is not exist E layer')
        minHmaxF[0, 0] = maxF
        minHmaxF[0, 1] = minH
    else:
        minHmaxF[0, 0] = np.min(r_location[0, :])
        minHmaxF[0, 1] = np.max(r_location[1, :])
    if g_location.shape[1] == 0:
        print('there is not exist F1 layer')
        minHmaxF[1, 0] = maxF
        minHmaxF[1, 1] = minH
    else:
        minHmaxF[1, 0] = np.min(g_location[0, :])
        minHmaxF[1, 1] = np.max(g_location[1, :])
    if b_location.shape[1] == 0:
        print('there is not exist F2 layer')
        minHmaxF[2, 0] = maxF
        minHmaxF[2, 1] = minH
    else:
        minHmaxF[2, 0] = np.min(b_location[0, :])
        minHmaxF[2, 1] = np.max(b_location[1, :])

    return minHmaxF
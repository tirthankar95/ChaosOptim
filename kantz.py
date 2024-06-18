s, r, b = 16, 15, 4

if __name__ == '__main__':
    def lorenz(data):
        x_ = s*(data[1]-data[0])
        y_ = r*data[0] - data[1] - data[0]*data[2]
        z_ = data[0]*data[1] - b*data[2]

    
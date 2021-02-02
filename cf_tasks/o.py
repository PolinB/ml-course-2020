if __name__ == '__main__':
    k = int(input())
    n = int(input())
    x_arr = []
    y_arr = []
    e_y2 = 0
    p_x = [0 for _ in range(k)]
    e_yx = [0 for _ in range(k)]
    for _ in range(n):
        x, y = map(int, input().split(" "))
        x_arr.append(x)
        y_arr.append(y)
        e_y2 += (y ** 2 / n)
        p_x[x - 1] += 1 / n
        e_yx[x - 1] += y / n
    ee = 0
    for i in range(k):
        if p_x[i] != 0:
            ee += e_yx[i] ** 2 / p_x[i]

    print(e_y2 - ee)

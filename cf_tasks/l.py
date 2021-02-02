import math
if __name__ == '__main__':
    sum_p = 0.
    sum_x = 0.
    sum_y = 0.
    sum_x_2 = 0.
    sum_y_2 = 0.
    n = int(input())
    for _ in range(n):
        x, y = map(float, input().split(" "))
        sum_p += x * y
        sum_x += x
        sum_x_2 += x ** 2
        sum_y += y
        sum_y_2 += y ** 2
    denum = (n * sum_x_2 - sum_x ** 2) * (n * sum_y_2 - sum_y ** 2)
    if denum < 0.0000001:
        print(0)
    else:
        print((n * sum_p - sum_x * sum_y) / math.sqrt(denum))

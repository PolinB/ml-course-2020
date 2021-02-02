big_neg_val = -1000
pos_val = 0.1
b_const = 0.02


def to_bool(x, m):
    res = [0 for _ in range(m)]
    cur = 0
    while x > 0:
        if x % 2 == 1:
            res[cur] = 1
        x = x // 2
        cur += 1
    return res


if __name__ == '__main__':
    m = int(input())
    f = []
    for _ in range(2 ** m):
        f.append(int(input()))

    print(2)
    print(2 ** m, 1)

    for i in range(2 ** m):
        b_vec = to_bool(i, m)
        # print(b_vec)

        b = b_const
        for b_val in b_vec:
            if b_val == 0:
                print(big_neg_val, end=" ")
            else:
                print(pos_val, end=" ")
                b -= pos_val
        print(b)

    for vv in f:
        print(vv, end=" ")
    print(-b_const)

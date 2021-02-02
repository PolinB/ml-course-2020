if __name__ == '__main__':
    k1, k2 = map(int, input().split(" "))
    n = int(input())
    x_arr = []
    y_arr = []
    ex = [0 for _ in range(k1)]
    ey = [0 for _ in range(k2)]
    cnt = {}
    mp = {}
    for _ in range(n):
        s = input()
        x, y = map(int, s.split(" "))
        ex[x - 1] += 1 / n
        ey[y - 1] += 1 / n
        if not cnt.__contains__(s):
            cnt[s] = 0
            mp[s] = (x - 1, y - 1)
        cnt[s] += 1
    res = n
    for v in cnt:
        ek = n * ex[mp[v][0]] * ey[mp[v][1]]
        res += ((cnt[v] - ek) ** 2 / ek - ek)
    print(res)

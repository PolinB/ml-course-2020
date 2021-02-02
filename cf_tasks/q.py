import math
if __name__ == '__main__':
    k1, k2 = map(int, input().split(" "))
    n = int(input())
    x_arr = []
    y_arr = []
    px = [0 for _ in range(k1)]
    cnt = {}
    mp = {}
    for _ in range(n):
        s = input()
        x, y = map(int, s.split(" "))
        px[x - 1] += 1 / n
        if not cnt.__contains__(s):
            cnt[s] = 0
            mp[s] = (x - 1, y - 1)
        cnt[s] += 1 / n
    res = 0
    for v in cnt:
        res += -cnt[v] * math.log(cnt[v] / px[mp[v][0]])
    print(res)

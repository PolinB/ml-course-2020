if __name__ == '__main__':
    k = int(input())
    n = int(input())
    k_arr = [[] for _ in range(k)]
    x_arr = []
    y_arr = []
    all_arr = []
    for _ in range(n):
        x, y = map(int, input().split(" "))
        y -= 1
        k_arr[y].append(x)
        x_arr.append(x)
        y_arr.append(y)
        all_arr.append((x, y))
    sum_in = 0
    sum_out = 0

    for k_i in range(len(k_arr)):
        cl = k_arr[k_i].copy()
        cl = sorted(cl)
        p_sum = 0
        s_sum = sum(cl)

        for i in range(len(cl)):
            s_sum -= cl[i]
            sum_in += (i * cl[i] - p_sum) + (s_sum - (len(cl) - 1 - i) * cl[i])
            p_sum += cl[i]
    print(sum_in)

    all_arr = sorted(all_arr)
    pref_sum = {y_i: 0 for y_i in range(k)}
    suf_sum = {y_i: 0 for y_i in range(k)}
    all_pref_sum = 0
    all_suf_sum = 0
    pref_cnt = {y_i: 0 for y_i in range(k)}
    suf_cnt = {y_i: 0 for y_i in range(k)}

    for i in range(n):
        all_suf_sum += x_arr[i]
        suf_sum[y_arr[i]] += x_arr[i]
        suf_cnt[y_arr[i]] += 1

    for i in range(n):
        x, y = all_arr[i][0], all_arr[i][1]
        all_pref_sum += x
        all_suf_sum -= x
        pref_sum[y] += x
        suf_sum[y] -= x
        pref_cnt[y] += 1
        suf_cnt[y] -= 1

        all_suf_cnt = n - (i + 1)
        sum_out += ((i + 1 - pref_cnt[y]) * x - (all_pref_sum - pref_sum[y]))
        sum_out += ((all_suf_sum - suf_sum[y]) - (all_suf_cnt - suf_cnt[y]) * x)
    print(sum_out)

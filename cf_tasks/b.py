eps = 0.000000000001


def safe_div(x, y):
    if abs(x) < eps:
        return 0.0
    return float(x / y)


def calc_f(matrix):
    k = len(matrix)
    sum_all = 0
    tp = [0 for _ in range(k)]
    fp = [0 for _ in range(k)]
    fn = [0 for _ in range(k)]
    tn = [0 for _ in range(k)]
    c = [0 for _ in range(k)]
    p = [0 for _ in range(k)]
    prec = [0. for _ in range(k)]
    recall = [0. for _ in range(k)]
    f = [0. for _ in range(k)]
    for i in range(k):
        for j in range(k):
            x = matrix[i][j]  # i - real class, j - act class
            sum_all += x
            c[i] += x
            p[j] += x
            if i == j:
                tp[i] = x
    for i in range(k):
        fp[i] = p[i] - tp[i]
        fn[i] = c[i] - tp[i]
        tn[i] = sum_all - tp[i] - fp[i] - fn[i]
        recall[i] = safe_div(tp[i], tp[i] + fn[i])
        prec[i] = safe_div(tp[i], tp[i] + fp[i])
        f[i] = safe_div(2.0 * prec[i] * recall[i], prec[i] + recall[i])

    micro_f = 0.
    prec_w = 0.
    recall_w = 0.
    for i in range(k):
        micro_f += (c[i] * f[i])
        prec_w += safe_div(tp[i] * c[i], p[i])
        recall_w += float(tp[i])
    micro_f /= float(sum_all)
    prec_w /= float(sum_all)
    recall_w /= float(sum_all)
    macro_f = safe_div(2.0 * prec_w * recall_w, prec_w + recall_w)
    return macro_f, micro_f


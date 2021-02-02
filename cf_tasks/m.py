if __name__ == '__main__':
    x_arr = []
    y_arr = []
    n = int(input())
    for _ in range(n):
        x, y = map(float, input().split(" "))
        x_arr.append(x)
        y_arr.append(y)
    x_arr_ind = [(x, i) for i, x in enumerate(x_arr)]
    y_arr_ind = [(y, i) for i, y in enumerate(y_arr)]
    x_arr_ind_sort = sorted(x_arr_ind)
    y_arr_ind_sort = sorted(y_arr_ind)

    d_arr = []
    for ind1 in range(len(x_arr_ind_sort)):
        x, i = x_arr_ind_sort[ind1]
        x_arr[i] = ind1
    for ind2 in range(len(y_arr_ind_sort)):
        y, j = y_arr_ind_sort[ind2]
        y_arr[j] = ind2
    for i in range(n):
        d_arr.append((x_arr[i] - y_arr[i]) ** 2)
    print(1 - (6 * sum(d_arr)) / (n * (n ** 2 - 1)))

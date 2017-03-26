from random import randint


def sort_and_count(array):
    """ Use Merge-Sort Idea to count inversions.

    :param array: target for inversions counting
    :return:
        inversion_count: count of inversion pair in array
        array_sorted: array sorted from small to large
    """

    size = len(array)

    # Stop condition of recursive call:
    # 1. When array only contains one element, number of inversions is 0
    #  and return itself.
    if size == 1:
        return 0, array
    # 2. Compare them when array contains two elements
    #  if contains a inversion, reverse two elements and number set to 1.
    if size == 2:
        if array[0] > array[1]:
            return 1, [array[1], array[0]]  # or array.reverse()
        else:
            return 0, array

    # Divide array into left and right parts from the middle.
    split_location = size / 2
    array_left = array[:split_location]
    array_right = array[split_location:]

    # Recursively call to sort_and_count left and right parts.
    count_left, array_left_sorted = sort_and_count(array_left)
    count_right, array_right_sorted = sort_and_count(array_right)

    # Merge after recursively call.
    count_merge, array_sorted \
        = merge_and_count(array_left_sorted, array_right_sorted)

    # Sum the inversion count of three parts.
    inversion_count = count_left + count_right + count_merge

    return inversion_count, array_sorted


def merge_and_count(array_x, array_y):
    """ Merge two sorted array with order and count the number of inversions

    :param array_x: sorted array from small to large order
    :param array_y: sorted array from small to large order
    :return:
        merge_inversion_count: number of inversions between array_x and array_y
        array_sorted: merge array_x and array_y into one array with order from
            small to large
    """
    size_x, size_y = len(array_x), len(array_y)
    index_x = index_y = 0
    merge_inversion_count = 0
    array_sorted = []

    while True:
        # In this situation, there is no element hasn't been compared
        #  in array_x, merge the elements reserved in array_y
        if index_x >= size_x:
            array_sorted += array_y[index_y:]
            break

        # Like the situation above.
        if index_y >= size_y:
            array_sorted += array_x[index_x:]
            break

        # Compare element of array_x and array_y and put smaller element into
        #  sorted array.
        # When put an element of array_y into array_sorted, it means every
        #  element reserved in array_x is bigger than it. Additional
        #  array_x_reserved_count inversions.
        if array_x[index_x] < array_y[index_y]:
            array_sorted.append(array_x[index_x])
            index_x += 1
        else:
            array_sorted.append(array_y[index_y])
            merge_inversion_count += size_x - index_x
            index_y += 1

    return merge_inversion_count, array_sorted


# Read file and return an array
def read_file(file_path):
    f = open(file_path, 'r')
    content = f.readlines()
    array = map(int, content)
    return array

# Run the algorithm
arr = read_file('./Q8.txt')
print sort_and_count(arr)[0]

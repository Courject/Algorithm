def karatsuba(x, y):
    """ Multiply two numbers with more efficient way than grade school's

    :param x: number one to multiply
    :param y: number two to multiply
    :return: product of x and y
    """

    # maximum length of x and y
    max_len = max(len(str(x)), len(str(y)))

    # stop condition of recursive call
    # Stop when one of the value equals 0 or their length equal 1
    if x == 0 or y == 0:
        return 0
    if max_len == 1:
        return x * y

    # Split x and y into high part and low part
    # It's easier for a human
    splitter = 10 ** (max_len / 2)  # Set the split location
    x_high = x // splitter
    x_low = x - x_high * splitter
    y_high = y // splitter
    y_low = y - y_high * splitter

    # parts will be used in product computing
    high_high = karatsuba(x_high, y_high)
    low_low = karatsuba(x_low, y_low)

    # The trick: grade school multiply twice. Only once in this way
    # One time multiply to get the result of x_high * y_low + x_low * y_high
    high_lows = (karatsuba(x_high + x_low, y_high + y_low)
                 - high_high - low_low)

    # Calculate the product of x and y by using computed parts
    result = (high_high * (splitter ** 2)
              + high_lows * splitter + low_low)

    return result


print "Input two number:"
num_a, num_b = map(int, raw_input().split())
print "Multiple Result: ", karatsuba(num_a, num_b)

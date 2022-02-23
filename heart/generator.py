

def example(n):

    num = 0
    while num < n:
        yield num
        num += 1


    


if __name__ == '__main__':
    lol = example(10)
    
    for i in lol:
        print(i)
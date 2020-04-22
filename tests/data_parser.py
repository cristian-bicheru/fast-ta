def parse(a):
    a = a.split("\n")
    a = [x+", " for x in a]
    a = [''.join(a[i:i+6])+"\n" for i in range(0, len(a), 6)]
    print(''.join(a))

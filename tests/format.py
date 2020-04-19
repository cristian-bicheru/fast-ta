data = """""".split("\n")

data = [str(i) for i in data]
for i,v in enumerate(data):
    a, b = v.split(".")
    a = " "*(3-len(a)) + a
    b = b + "0"*(6-len(b))
    data[i] = a + "." + b
n = 6
for i in range(0, len(data), n):
    print(", ".join(data[i:i+n]),end=",\n")


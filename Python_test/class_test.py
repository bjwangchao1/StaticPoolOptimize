class Test(object):
    def __init__(self):
        self.__num = 100

    def setNum(self, newNum):
        self.__num = newNum

    def getNum(self):
        return self.__num

    num = property(getNum, setNum)


t = Test()
# t.__num = 200
# print(t.__num)

print(t.getNum())
t.setNum(50)
print(t.getNum())

print('-' * 50)

t.num = 400
print(t.num)

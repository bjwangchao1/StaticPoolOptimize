from multiprocessing import Process
import time


def f(n):
    time.sleep(1)
    print(n * n)


if __name__ == '__main__':
    for i in range(10):
        P = Process(target=f, args=[i, ])
        P.start()
        # P.join()
        # P.terminate() # 关闭

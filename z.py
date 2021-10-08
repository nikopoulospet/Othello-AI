import multiprocessing
from time import sleep

from src.Agents.agentThreaded import findMin
class MyFancyClass(object):
    
    def __init__(self, name):
        self.name = name
    
    def do_something(self):
        proc_name = multiprocessing.current_process().name
        print ('Doing something fancy in %s for %s!' % (proc_name, self.name))


def worker(q:multiprocessing.Queue, eventName):
    while not eventName.is_set():
        pass
    obj = q.get()
    obj.do_something()


event = multiprocessing.Event()

if __name__ == '__main__':
    queue = multiprocessing.Queue()
    queueList = [multiprocessing.Queue()] * 3
    queueList[0].put(1)
    p = multiprocessing.Process(target=worker, args=(queue,event))
    p.start()
    if queueList:
        print("QUELIST")
    if not queueList:
        print("NOT QUELIST")
    sleep(1)
    queue.put(MyFancyClass('Fancy Dan'))
    queue.put(MyFancyClass('Fancy Dan2'))
    queue.put(MyFancyClass('Fancy Dan3'))
    sleep(1)
    print("setting event")
    event.set()
    # Wait for the worker to finish
    queue.close()
    queue.join_thread()
    p.join()
    
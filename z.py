from threading import Thread, Event
from time import sleep

event = Event()
TEST = "FAILED"
def modify_variable(var):
    while True:
        for i in range(len(var)):
            var[i] += 1
        if check():
            break
        sleep(.5)
    print('Stop printing')

def check():
    TEST = "PASS"
    return event.is_set()



my_var = [1, 2, 3]
t = Thread(target=modify_variable, args=(my_var, ))
t.start()
while True:
    try:
        print(my_var)
        sleep(1)
    except KeyboardInterrupt:
        event.set()
        break
t.join()
print(TEST)
print(my_var)
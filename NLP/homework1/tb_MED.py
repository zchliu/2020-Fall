import MED
import random
import string
import time

time_start = time.time()

for i in range(100):
    ran_str1 = ''.join(random.sample(string.ascii_letters + string.digits, 50))
    ran_str2 = ''.join(random.sample(string.ascii_letters + string.digits, 50))

    distance, D = MED.edit(ran_str1, ran_str2)
    path = MED.bfs(D[len(ran_str1), len(ran_str2)], D[0, 0])
    print("The minimun editting distance is: ", distance)
    MED.output(ran_str1, ran_str2, distance, D, path)

time_end = time.time()
print('average cost', (time_end - time_start) / 100)

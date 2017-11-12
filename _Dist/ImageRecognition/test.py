from ProgressBar import ProgressBar
import time

def task(cost=0.5, epoch=3, name="", sub_task=None):
    def sub():
        bar = ProgressBar(max_value=epoch, name=name)
        for _ in range(epoch):
            time.sleep(cost)
            if sub_task is not None:
                sub_task()
            bar.update()
    return sub

task(name="Task1", sub_task=task(
    name="Task2", sub_task=task(
        name="Task3")))()



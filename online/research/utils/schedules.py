'''
This file contains schedule functions that can be used as learning rate schedules

All learning rate schedulers use the pytorch LambdaLR function and any additional kwargs.
'''

def linear_decay(total_steps, start_step=1, offset=0):

    def fn(step):
        return 1.0 - max(0, step + offset - start_step) / (total_steps - start_step)

    return fn

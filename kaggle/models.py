import datetime


class LogMessage:
    def __init__(self, task, pid, iteration, eta, norm_grad, norm_beta, objective, training_error, accuracy):
        self.task = task
        self.pid = pid
        self.iteration = iteration
        self.eta = eta
        self.norm_grad = norm_grad
        self.norm_beta = norm_beta
        self.objective = objective
        self.accuracy = accuracy
        self.training_error = training_error
        self.timestamp = datetime.datetime.now() 

    def __str__(self):
        return "%s %s %s %s %s %s %s %s %s" % (self.timestamp,
                                               self.pid,
                                               self.task,
                                               self.iteration,
                                               self.eta,
                                               self.norm_grad,
                                               self.objective,
                                               self.accuracy,
                                               self.training_error)

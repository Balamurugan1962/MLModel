from .reid_tester import ReIDTester
from core.solvers.solver_multitask_dev import TesterMultiTaskDev

def tester_entry(C_train, C_test):
    return globals()[C_test.config['common']['tester']['type']](C_train, C_test)

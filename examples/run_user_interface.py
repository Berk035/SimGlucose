from simglucose.simulation.user_interface import simulate
import unittest
from unittest.mock import patch
import shutil
import os, inspect

parentdir = os.path.join(os.path.expanduser("~"),'PycharmProjects','simglucose')
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
output_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           '..', '..', 'examples', 'results'))
print(output_folder)

# animation, parallel, save_path, sim_time, scenario, scenario random
# seed, start_time, patients, sensor, sensor seed, insulin pump,
# controller
# mock_input=patch('builtins.input')
# mock_input.side_effect = ['y', 'n', output_folder, '24', '1', '2','6', '5', '1', 'd', '1', '1', '2', '1']

simulate()

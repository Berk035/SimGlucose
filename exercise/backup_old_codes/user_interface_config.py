from simglucose.simulation.sim_engine import SimObj, batch_sim
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.controller.pid_ctrller import PIDController

#TODO: RL controller will implemented it
from simglucose.controller.rl_ctrller import RLController

from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.scenario import CustomScenario
from simglucose.analysis.report import report
import pandas as pd
import copy
import pkg_resources
import logging
import os, inspect, yaml
from datetime import datetime
from datetime import timedelta
import platform


logger = logging.getLogger(__name__)

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')
SENSOR_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/sensor_params.csv')
INSULIN_PUMP_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/pump_params.csv')

def parse_config(config):
    with open(config, 'r') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    return config_data

parentdir = os.path.join(os.path.expanduser("~"),'VS_Project','simglucose')
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

config_file = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           '..', '..', 'examples', 'configs', 'user_info.yaml'))
print(config_file)
config = parse_config(config_file)

def pick_patients(select1=None,select2=None):
    patient_params = pd.read_csv(PATIENT_PARA_FILE)
    while (select1 is None):
        select1 = input('Select virtual patients:\n' +
                        '[1] All\n' +
                        '[2] All Adolescents\n' +
                        '[3] All Adults\n' +
                        '[4] All Children\n' +
                        '[5] By ID\n' +
                        '>>> ')
        try:
            select1 = int(select1)
        except ValueError:
            print('Please input an integer. Try again')
            input('Press any key to continue ...')
            continue

        if select1 < 1 or select1 > 5:
            print('Input 1 to 5 please!')
            input('Press any key to continue ...')
            continue
        else:
            break

    if select1 == 1:
        patients = patient_params['Name']
    elif select1 == 2:
        patients = patient_params['Name'][0:10]
    elif select1 == 3:
        patients = patient_params['Name'][10:20]
    elif select1 == 4:
        patients = patient_params['Name'][20:30]
    else:
        patients = []
        select_hist = []
        while True:
            print('Select patient:')
            for j in range(len(patient_params)):
                print('[{0}] {1}'.format(j + 1, patient_params['Name'][j]))
            print('[D] Done')
            select2 = input('>>> ')

            if select2 == 'D' or select2 == 'd':
                break

            try:
                select2 = int(select2)
            except ValueError:
                print("Please input a number or 'D' or 'd'.")
                input('Press any key to continue ...')
                continue

            if select2 < 1 or select2 > 30:
                print("Please input an number from 1 to {0}.".format(
                    len(patient_params)))
                input('Press any key to continue ...')
                continue

            if select2 in select_hist:
                print("{0} is already selected!".format(
                    patient_params['Name'][select2 - 1]))
                input('Press any key to continue ...')
                continue
            else:
                select_hist.append(select2)
                patients.append(patient_params['Name'][select2 - 1])
    logger.info('Selected patients:\n{}'.format(patients))
    return patients


def pick_cgm_sensor(cgm=None, s_seed=None):
    sensor_params = pd.read_csv(SENSOR_PARA_FILE)
    total_sensor_num = len(sensor_params.index)
    selection, seed = cgm, s_seed
    while (cgm is None):
        print('Select the CGM sensor:')
        for i in range(total_sensor_num):
            print('[{0}] {1}'.format(i + 1, sensor_params['Name'][i]))
        input_value = input('>>> ')
        try:
            selection = int(input_value)
        except ValueError:
            print("Oops! Please input a number.")
            input('Press any key to continue ...')
            continue
        if selection < 1 or selection > total_sensor_num:
            print("Please input an integer from 1 to {0}!".format(
                total_sensor_num))
            input('Press any key to continue ...')
            continue
        else:
            break
    sensor = sensor_params['Name'][selection - 1]
    logger.info('Selected sensor:\n{}'.format(sensor))

    while (s_seed is None):
        input_value = input('Select Random Seed for Sensor Noise [None]: ')
        try:
            seed = int(input_value)
            break
        except ValueError:
            if input_value == '' or input_value == 'None':
                seed = None
                break
            else:
                print('Please input an integer!')
                continue
    logger.info('Sensor Random Seed: {}'.format(seed))
    return sensor, seed


def pick_insulin_pump(name_pump=None):
    pump_params = pd.read_csv(INSULIN_PUMP_PARA_FILE)
    selection = name_pump
    while (name_pump is None):
        print('Select the insulin pump:')
        for i in range(len(pump_params)):
            print('[{}] {}'.format(i + 1, pump_params['Name'][i]))
        input_value = input('>>> ')
        try:
            selection = int(input_value)
        except ValueError:
            print("Oops! Please input a number.")
            input('Press any key to continue ...')
            continue
        if selection < 1 or selection > len(pump_params):
            print("Please input an integer from 1 to {0}!".format(
                len(pump_params)))
            input('Press any key to continue ...')
            continue
        else:
            break
    pump = pump_params['Name'][selection - 1]
    logger.info('Selected Pumps:\n{}'.format(pump))
    return pump


def pick_scenario():
    while True:
        print('Select scnenario:')
        print('[1] Random Scnenario')
        print('[2] Custom Scnenario')
        input_value = input('>>>')
        try:
            selection = int(input_value)
        except ValueError:
            print('Please input an integer!')
            continue
        if selection < 1 or selection > 2:
            print('Please input a number from the list!')
        else:
            break
    if selection == 1:
        while True:
            input_value = input(
                'Select random seed for random scenario [None]: ')
            try:
                seed = int(input_value)
                break
            except ValueError:
                if input_value == '' or input_value == 'None':
                    seed = None
                    break
                else:
                    print('Please input an integer!')
                    continue
        scenario = RandomScenario(seed=seed)
    elif selection == 2:
        scenario = CustomScenario()

    return scenario


def pick_controller(cnt=None):
    selection = cnt
    while (cnt is None):
        print('Select controller:')
        print('[1] Basal-Bolus Controller')
        print('[2] PID Controller')
        print('[3] RL Controller')
        input_value = input('>>>')
        try:
            selection = int(input_value)
        except ValueError:
            print('Please input an integer!')
            continue
        if selection < 1 or selection > 3:
            print('Please input a number from the list!')
        else:
            break
    if selection == 1:
        controller = BBController()
    elif selection == 2:
        controller = PIDController()
    elif selection == 3:
        controller = RLController()
    else:
        print("Controller error!")
        controller = None
    return controller


def build_envs(scenario, start_time):
    sel = config["patients"]
    cgm = config["sensor"]
    cgm_seed = config["sensor_seed"]
    pump = config["insulin_pump"]

    patient_names = pick_patients(select1=sel)
    cgm_sensor_name, cgm_seed = pick_cgm_sensor(cgm=cgm, s_seed=cgm_seed)
    insulin_pump_name = pick_insulin_pump(name_pump=pump)
    if scenario is None:
        scenario = pick_scenario()

    def local_build_env(pname):
        patient = T1DPatient.withName(pname)
        cgm_sensor = CGMSensor.withName(cgm_sensor_name, seed=cgm_seed)
        insulin_pump = InsulinPump.withName(insulin_pump_name)
        scen = copy.deepcopy(scenario)
        env = T1DSimEnv(patient, cgm_sensor, insulin_pump, scen)
        return env

    envs = [local_build_env(p) for p in patient_names]
    return envs


def pick_save_path():
    foldername = input('Folder name to save results [default]: ')
    if foldername == 'default' or foldername == '':
        foldername = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    save_path = os.path.join(os.path.abspath('./results/'), foldername)
    print('Results will be saved in {}'.format(save_path))
    return save_path


def create_sim_instance(sim_time=None,
                        scenario=None,
                        controller=None,
                        start_time=None,
                        save_path=None,
                        animate=True):

    sim_time = timedelta(hours=float(config["sim_duration"]))
    if sim_time is None:
        sim_time = timedelta(hours=float(
            input('Input simulation time (hr): ')))

    scenario = config["scenario"]
    seed = config["scenario_seed"]
    st = config["start_time"]
    if scenario == 1:
        scenario = RandomScenario(seed=seed, start_time=st)
    elif scenario == 2:
        scenario = CustomScenario()
    elif scenario is None:
        scenario = pick_scenario()
    envs = build_envs(scenario, start_time)

    controller = config["controller"]

    if controller == 1:
        controller = BBController()
    elif controller == 2:
        controller = PIDController()
    elif controller == 3:
        controller = RLController()
    elif controller is None:
        controller = pick_controller()

    ctrllers = [copy.deepcopy(controller) for _ in range(len(envs))]

    sim_instances = [SimObj(e,
                            c,
                            sim_time,
                            animate=animate,
                            path=save_path) for (e, c) in zip(envs, ctrllers)]
    return sim_instances


def simulate(sim_time=None,
             scenario=None,
             controller=None,
             start_time=None,
             save_path=None,
             animate=None,
             parallel=None):
    '''
    Main user interface.
    ----
    Inputs:
    sim_time   - a datetime.timedelta object specifying the simulation time.
    scenario   - a simglucose.scenario.Scenario object. Use
                 simglucose.scenario_gen.RandomScenario or
                 simglucose.scenario.CustomScenario to create a scenario object.
    controller - a simglucose.controller.Controller object.
    start_time - a datetime.datetime object specifying the simulation start time.
    save_path  - a string representing the directory to save simulation results.
    animate    - switch for animation. True/False.
    parallel   - switch for parallel computing. True/False.
    '''

    animate = config["show_animation"]
    parallel = config["multiple_process"]

    if config["result_folder_name"] is None:
        save_path = pick_save_path()
    else:
        save_path = '/home/berk/VS_Project/simglucose/examples/results/' + str(config["result_folder_name"])

    if animate is None:
        while True:
            select = input('Show animation? (y/n) ')
            if select == 'y':
                animate = True
                break
            elif select == 'n':
                animate = False
                break
            else:
                continue

    if parallel is None:
        while True:
            select = input('Use multiple processes? (y/n) ')
            if select == 'y':
                parallel = True
                break
            elif select == 'n':
                parallel = False
                break
            else:
                continue

    if platform.system() == 'Darwin':
        if animate is True and parallel is True:
            raise ValueError(
                """animate and parallel cannot be turned on at the same time in macOS.""")

    sim_instances = create_sim_instance(sim_time=sim_time,
                                        scenario=scenario,
                                        controller=controller,
                                        start_time=start_time,
                                        save_path=save_path,
                                        animate=animate)
    results = batch_sim(sim_instances, parallel=parallel)

    df = pd.concat(results, keys=[s.env.patient.name for s in sim_instances])
    results, ri_per_hour, zone_stats, figs, axes = report(df, save_path)

    return 0


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # logger.setLevel(logging.INFO)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    # ch.setLevel(logging.DEBUG)
    ch.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter(
        '%(process)d: %(name)s: %(levelname)s: %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    root.addHandler(ch)

    simulate()

__all__ = []

import os
import sys
import random
import time
import shutil
import argparse
import subprocess
from pathlib import Path
import yaml


# =========================================================
# Helper functions
# =========================================================


def yaml_dump(obj, path):
    with Path(path).open('w') as f:
        yaml.dump(obj, f, default_flow_style=False, indent=4, width=9999)


def yaml_load(path):
    with Path(path).open('r') as f:
        d = yaml.load(f, Loader=yaml.FullLoader)
    return d


def get_full_command(argv):
    new_argv = []
    for index, arg in enumerate(argv):
        if index == 0:
            new_argv.append(Path(argv[0]).name)
        elif ' ' in arg:
            new_argv.append('"{}"'.format(arg))
        else:
            new_argv.append(arg)
    return ' '.join(new_argv)


def fail(*args):
    print('Fail:\n\t{}'.format('\n\t'.join(args)), flush=True)
    sys.exit()


def success(*args):
    print('Success:\n\t{}'.format('\n\t'.join(args)), flush=True)


def trace_up(target_file):
    path = Path.cwd()
    while not path.joinpath(target_file).is_file():
        if path == Path('/'):
            return None
        path = path.parent
    return path / target_file


def find_hogwarts(should_exist):
    hogwarts_file = trace_up('.hogwarts')
    do_exist = hogwarts_file is not None
    if do_exist != should_exist:
        if do_exist:
            fail('hogwarts is already built at {}.'.format(hogwarts_file))
        else:
            fail('hogwarts is not found.')
    return hogwarts_file


def find_house(house, should_exist):
    hogwarts_file = find_hogwarts(True)
    hogwarts_info = yaml_load(hogwarts_file)
    if house == '':
        house = hogwarts_info['curr_house']
    do_exist = house in hogwarts_info['avail_houses']
    if do_exist:
        house_dir = hogwarts_file.parent / hogwarts_info['avail_houses'][house]
        house_file = house_dir / '.house'
        do_exist = house_dir.is_dir() and house_file.is_file()
    if do_exist != should_exist:
        if do_exist:
            fail('house {} is already built at {}.'.format(house, house_file))
        else:
            fail('house {} is not found.'.format(house))
    return house_file if do_exist else None


def find_wizard(wizard, should_exist):
    if wizard == '':
        wizard_file = trace_up('.wizard')
        do_exist = wizard_file is not None
    else:
        house_file = find_house('', True)
        wizard_dir = house_file.parent / wizard
        wizard_file = wizard_dir / '.wizard'
        do_exist = wizard_dir.is_dir() and wizard_file.is_file()
    if do_exist != should_exist:
        if do_exist:
            fail('wizard already exists at {}.'.format(wizard_file))
        else:
            fail('wizard is not found.')
    return wizard_file if do_exist else None


def build_hogwarts():
    find_hogwarts(False)
    hogwarts_file = Path.cwd() / '.hogwarts'
    hogwarts_info = {
        'curr_house': '',
        'avail_houses': {}}
    yaml_dump(hogwarts_info, hogwarts_file)
    success('built hogwarts at {}'.format(hogwarts_file))


def build_house():
    hogwarts_file = find_hogwarts(True)
    hogwarts_info = yaml_load(hogwarts_file)
    find_house(Path.cwd().name, False)
    house_file = Path.cwd() / '.house'
    yaml_dump({}, house_file)
    prev_house = hogwarts_info['curr_house']
    hogwarts_info['curr_house'] = Path.cwd().name
    hogwarts_info['avail_houses'][hogwarts_info['curr_house']] = \
        str(house_file.parent.relative_to(hogwarts_file.parent))
    yaml_dump(hogwarts_info, hogwarts_file)
    success('built house at {}'.format(house_file),
            'switched house from {} to {}'.format(
                prev_house, hogwarts_info['curr_house']))


def cd_and_execute(log_dir, trg_dir, command, wizard, hrank):
    env = os.environ.copy()
    env['wizard'] = str(wizard)
    env['log_dir'] = str(log_dir)
    env['hrank'] = str(hrank)
    process = subprocess.Popen(command, cwd=str(trg_dir.resolve()), shell=True, env=env)
    return process


# =========================================================
# Shell script interfaces
# =========================================================


def control():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--build', '-b', action="store_true",
                       help='build hogwarts/house')
    group.add_argument('--switch', '-s', action="store_true",
                       help='switch current house')
    group.add_argument('--delete', '-d', action='store_true',
                       help='delete a house')
    parser.add_argument('name')
    opt = parser.parse_args()
    if opt.build:
        if opt.name.casefold() == 'hogwarts':
            build_hogwarts()
        elif opt.name.casefold() == 'house':
            build_house()
        else:
            fail('unexpected building: {} (hogwarts/house expected)'.format(opt.name))
    elif opt.switch:
        hogwarts_file = find_hogwarts(True)
        hogwarts_info = yaml_load(hogwarts_file)
        if opt.name in hogwarts_info['avail_houses']:
            prev_house = hogwarts_info['curr_house']
            hogwarts_info['curr_house'] = opt.name
            yaml_dump(hogwarts_info, hogwarts_file)
            success('switched house from {} to {}'.format(prev_house,
                                                          hogwarts_info['curr_house']))
        else:
            fail('unexpected house: {}'.format(opt.name),
                 'available houses: {}'.format('/'.join(hogwarts_info['avail_houses'].keys())))
    elif opt.delete:
        hogwarts_file = find_hogwarts(True)
        hogwarts_info = yaml_load(hogwarts_file)
        if opt.name in hogwarts_info['avail_houses']:
            del hogwarts_info['avail_houses'][opt.name]
            if opt.name == hogwarts_info['curr_house']:
                hogwarts_info['curr_house'] = ''
            yaml_dump(hogwarts_info, hogwarts_file)
            success('deleted house {}, current house is {}'.format(opt.name,
                                                                   hogwarts_info['curr_house']))
        else:
            fail('unexpected house: {}'.format(opt.name),
                 'available houses: {}'.format('/'.join(hogwarts_info['avail_houses'].keys())))


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--resume', '-r', action='store_true', default=False)
    parser.add_argument('--force', '-f', action='store_true', default=False)
    parser.add_argument('--command', '-c')
    parser.add_argument('--hsize', '-s', type=int, default=1)
    parser.add_argument('--parallel', '-p', action='store_true', default=False)
    opt = parser.parse_args()

    assert opt.hsize > 0, 'world size smaller than 1!'
    os.environ['hsize'] = str(opt.hsize)

    hogwarts_file = find_hogwarts(True)
    house_file = find_house('', True)

    src_dir = Path.cwd()
    if src_dir == hogwarts_file.parent:
        fail('curr directory contains .hogwarts.')
    if str(house_file).startswith(str(src_dir)):
        fail('curr directory contains {}.'.format(house_file.relative_to(src_dir)))

    wizard_dir = house_file.parent / opt.name
    force, resume = opt.force, opt.resume
    while wizard_dir.is_dir():
        if force:
            shutil.rmtree(str(wizard_dir))
            break
        elif resume:
            break
        else:
            print('wizard {} already exist at {}, '
                  'overwrite/resume/break? [Y/r/n] '.format(repr(opt.name), wizard_dir), end='', flush=True)
            choice = input().strip().casefold()
            if choice == 'y':
                force = True
            elif choice == 'r':
                resume = True
            elif choice == 'n':
                sys.exit()

    random.seed(42)
    if resume:
        wizard_file = find_wizard(opt.name, True)
        wizard = str(wizard_file.parent.relative_to(house_file.parent))
        runway_info = yaml_load(wizard_file)
        trg_dir = wizard_file.parent / runway_info['trg_dir_from_wizard']
        processes = []
        for hrank in range(opt.hsize):
            hrank = random.randint(0, 1000000)
            log_dir = wizard_dir / '{:d}'.format(hrank)
            log_dir.mkdir(parents=True, exist_ok=True)
            wizard = '{}/{:d}'.format(opt.name, hrank)
            process = cd_and_execute(log_dir, trg_dir, runway_info['sub_command'], wizard, hrank)
            processes.append(process)
            if not opt.parallel:
                try:
                    while True:
                        process.wait()
                        break
                except KeyboardInterrupt:
                    print('\tPlease double press Ctrl-C within 1 second to kill job.'
                          'It will take several seconds to shutdown ...', flush=True)
                    break
        if opt.parallel:
            try:
                for process in processes:
                    process.wait()
            except KeyboardInterrupt:
                for process in processes:
                    process.kill()
    else:
        if opt.command is None:
            raise argparse.ArgumentError(None, 'command required')
        wizard_dir.mkdir(parents=True, exist_ok=True)
        wizard_file = wizard_dir / '.wizard'
        trg_dir = wizard_dir / src_dir.name
        runway_info = {
            'date': time.strftime('%Y-%m-%d-%H:%M:%S'),
            'src_dir_from_hogwarts': str(src_dir.relative_to(hogwarts_file.parent)),
            'trg_dir_from_wizard': src_dir.name,
            'sub_command': opt.command,
            'full_command': get_full_command(sys.argv),
        }
        yaml_dump(runway_info, wizard_file)
        shutil.copytree(str(src_dir), str(wizard_dir / src_dir.name))
        processes = []
        for hrank in range(opt.hsize):
            hrank = random.randint(0, 1000000)
            log_dir = wizard_dir / '{:d}'.format(hrank)
            log_dir.mkdir(parents=True, exist_ok=True)
            wizard = '{}/{:d}'.format(opt.name, hrank)
            process = cd_and_execute(log_dir, trg_dir, opt.command, wizard, hrank)
            processes.append(process)
            if not opt.parallel:
                try:
                    while True:
                        process.wait()
                        break
                except KeyboardInterrupt:
                    print('\tPlease double press Ctrl-C within 1 second to kill job.'
                          'It will take several seconds to shutdown ...', flush=True)
                    break
        if opt.parallel:
            try:
                for process in processes:
                    process.wait()
            except KeyboardInterrupt:
                for process in processes:
                    process.kill()


def ls():
    print('hogwarts: {}'.format(find_hogwarts(True).parent), flush=True)
    print('house:    {}'.format(find_house('', True).parent), flush=True)


# =========================================================
# Test
# =========================================================


if __name__ == '__main__':
    success('hogwarts', 'gryffindor')

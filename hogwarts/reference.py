__all__ = []

import os
import sys
import time
import yaml
import shlex
import shutil
import argparse
import subprocess
from pathlib import Path

#################### helper functions ####################

def yaml_dump(obj, file):
    with Path(file).open('w') as f:
        yaml.dump(obj, f, default_flow_style = False, indent = 4, width = 9999)

def yaml_load(file):
    with Path(file).open('r') as f:
        d = yaml.load(f)
    return d

def get_full_command(argv):
    new_argv = []
    for index, arg in enumerate(argv):
        if index == 0:
            new_argv.append(Path(argv[0]).name)
        elif ' ' in arg:
            new_argv.append('"' + arg + '"')
        else:
            new_argv.append(arg)
    return ' '.join(new_argv)

def fail(msg):
    print('Fail:\n\t' + msg, flush = True)
    sys.exit()

def success(msg):
    print('Success:\n\t' + msg, flush = True)

def trace_up(target_file):
    dir = Path.cwd()
    while dir != Path('/') and not (dir / target_file).is_file():
        dir = dir.parent
    path = dir / target_file
    do_exist = path.is_file()
    if path.is_file():
        return path
    else:
        return None

def find_school(should_exist):
    school_file = trace_up('.pg-school')
    do_exist = school_file is not None
    if do_exist != should_exist:
        if do_exist:
            fail('pg school already initialized at %s.' % school_file)
        else:
            fail('pg school not found.')
    return school_file

def find_playground(should_exist):
    school_file = find_school(True)
    school_info = yaml_load(school_file)
    do_exist = school_info['playground_relto_school'] != ''
    playground_file = school_file.parent / school_info['playground_relto_school'] / '.pg-playground'
    if do_exist != should_exist:
        if do_exist:
            fail('pg playground already intialized at %s' % playground_file)
        else:
            fail('pg playground not found.')
    return playground_file if do_exist else None

def find_runway(run_name, should_exist):
    if run_name == '':
        runway_file = trace_up('.pg-runway')
        do_exist = runway_file is not None
    else:
        playground_file = find_playground(True)
        runway_dir = playground_file.parent / run_name
        runway_file = runway_dir / '.pg-runway'
        do_exist = runway_dir.is_dir() and runway_file.is_file()
    if do_exist != should_exist:
        if do_exist:
            fail('pg runway already exist at %s.' % runway_file)
        else:
            fail('pg runway not found.')
    return runway_file if do_exist else None

def init_school():
    find_school(False)
    school_file = Path.cwd() / '.pg-school'
    school_info = { 'playground_relto_school': '' }
    yaml_dump(school_info, school_file)
    success('initialize pg school at %s' % school_file)

def init_playground():
    school_file = find_school(True)
    find_playground(False)
    playground_file = Path.cwd() / '.pg-playground'
    yaml_dump({}, playground_file)
    school_info = {
        'playground_relto_school': str(playground_file.parent.relative_to(school_file.parent)),
    }
    yaml_dump(school_info, school_file)
    success('initialize playground at %s' % playground_file)

def cd_and_execute(trg_dir, command, run_name):
    os.chdir(str(trg_dir))
    env = os.environ.copy()
    env['run_name'] = run_name
    process = subprocess.Popen(command, shell = True, env = env)
    while True:
        try:
            process.wait()
            break
        except KeyboardInterrupt:
            print('\tPlease double press Ctrl-C within 1 second to kill srun job. It will take several seconds to shutdown ...', flush = True)

#################### command line script interfaces ####################

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', '-rn', default = 'default')
    parser.add_argument('--force', '-f', action = 'store_true', default = False)
    parser.add_argument('--command', '-c', required = True)
    opt = parser.parse_args()

    school_file = find_school(True)
    playground_file = find_playground(True)

    src_dir = Path.cwd()
    if src_dir == school_file.parent:
        fail('current directory contains .pg-school.')
    if str(playground_file).startswith(str(src_dir)):
        fail('current directory contains %s.' % playground_file.relative_to(src_dir))

    runway_dir = playground_file.parent / opt.run_name
    if runway_dir.is_dir():
        while not opt.force:
            print('runway %s already exist at %s, overwrite or not? [Y/n] ' % (
                repr(opt.run_name), runway_dir,
            ), end = '')
            Yn = input().strip()
            if Yn in ['Y', 'y']:
                break
            elif Yn in ['N', 'n']:
                sys.exit()
            else:
                continue
        shutil.rmtree(str(runway_dir))

    runway_dir.mkdir(parents = True, exist_ok = True)
    runway_file = runway_dir / '.pg-runway'
    trg_dir = runway_dir / src_dir.name
    runway_info = {
        'date': time.strftime('%Y-%m-%d-%H:%M:%S'),
        'src_dir_relto_school': str(src_dir.relative_to(school_file.parent)),
        'trg_dir_relto_runway': src_dir.name,
        'sub_command': opt.command,
        'full_command': get_full_command(sys.argv),
    }
    yaml_dump(runway_info, runway_file)
    shutil.copytree(str(src_dir), str(trg_dir))
    cd_and_execute(trg_dir, opt.command, opt.run_name)

def resume():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', '-rn', default = '')
    opt = parser.parse_args()
    playground_file = find_playground(True)
    runway_file = find_runway(opt.run_name, True)
    run_name = str(runway_file.parent.relative_to(playground_file.parent))
    runway_info = yaml_load(runway_file)
    trg_dir = runway_file.parent / runway_info['trg_dir_relto_runway']
    cd_and_execute(trg_dir, runway_info['sub_command'], run_name)

def reproduce():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_run_name', '-srn', default = '')
    parser.add_argument('--trg_run_name', '-trn', default = 'default')
    parser.add_argument('--force', '-f', action = 'store_true', default = False)
    opt = parser.parse_args()
    runway_file = find_runway(opt.src_run_name, True)

    runway_info = yaml_load(runway_file)
    trg_dir = runway_file.parent / runway_info['trg_dir_relto_runway']
    os.chdir(str(trg_dir))
    psudo_command = 'pg-run --run_name %s %s-c %s' % (
        opt.trg_run_name,
        '-f ' if opt.force else '',
        repr(runway_info['sub_command']),
    )
    sys.argv = shlex.split(psudo_command)
    run()

def clean():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', '-rn', default = '')
    opt = parser.parse_args()
    runway_file = find_runway(opt.run_name, True)

    runway_info = yaml_load(runway_file)
    trg_dir_name = runway_info['trg_dir_relto_runway']
    for subpath in runway_file.parent.glob('*'):
        if subpath.name not in ['.pg-runway', trg_dir_name]:
            if subpath.is_dir():
                shutil.rmtree(str(subpath))
            else:
                os.remove(str(subpath))

def destroy():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', '-rn', default = '')
    opt = parser.parse_args()
    runway_file = find_runway(opt.run_name, True)

    shutil.rmtree(str(runway_file.parent))

def list_runway():
    parser = argparse.ArgumentParser()
    parser.add_argument('--order', '-o', default = 'date',
            help = 'list order ("date" / "name"), default "date"')
    parser.add_argument('--command', '-c', action = 'store_true', default = False,
            help = 'list sub command, default False')
    parser.add_argument('--full_command', '-fc', action = 'store_true', default = False,
            help = 'list full command, default False')
    opt = parser.parse_args()
    assert opt.order in ['date', 'name', '']

    playground_file = find_playground(True)
    runways = []
    for dirpath, dirnames, filenames in os.walk(str(playground_file.parent), topdown = True):
        if '.pg-runway' in filenames:
            dirnames.clear()
            name = str(Path(dirpath).relative_to(playground_file.parent))
            runway_file = Path(dirpath) / '.pg-runway'
            runway_info = yaml_load(runway_file)
            runways.append((runway_info['date'], name, runway_info))
    if opt.order == 'date':
        runways.sort()
    elif opt.order == 'name':
        runways.sort(key = lambda x: x[1])
    maxlen = max(len(name) for _, name, _ in runways)
    for index, (date, name, runway_info) in enumerate(runways):
        if opt.full_command:
            print('(%d) %s' % (index, runway_info['full_command']), flush = True)
        elif opt.command:
            name = ('%%-%ds' % maxlen) % name
            print('(%d) [%s] %s | %s' % (index, date, name, runway_info['sub_command']), flush = True)
        else:
            print('(%d) [%s] %s' % (index, date, name), flush = True)


def school_dir():
    print(find_school(True).parent)


def playground_dir():
    print(find_playground(True).parent)


def runway_dir():
    print(find_runway('', True).parent)


#################### training script interfaces ####################

## def init_pg():
##     school_file = find_school(True)
##     runway_file = find_runway('', True)
##     with runway_file.open('r') as f:
##         relative_src_dir = Path(f.readline().strip())
##     src_dir = school_file.parent / relative_src_dir
##     trg_dir = runway_file.parent / relative_src_dir.name
##     os.environ['pg_src_dir'] = str(src_dir)
##     os.environ['pg_trg_dir'] = str(trg_dir)
##
## def pg_path(path):
##     if 'pg_src_dir' not in os.environ:
##         raise Exception('pg not initialized yet.')
##     path = Path(path).absolute()
##     path = Path(os.path.realpath(str(path)))
##     if path.startswith(os.environ['pg_src_dir']):
##         relative_path = path.relative_to(os.environ['pg_src_dir'])
##         path = Path(os.environ['pg_trg_dir']) / relative_path
##     return path

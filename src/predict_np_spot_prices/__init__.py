import argparse
import os
import sys
import traceback
from predict_np_spot_prices.common import (
    DATA_DIR_PREPROCESSED,
    DATA_DIR_SPECIAL,
    DataCategory,
    get_area_code,
    get_timestamp,
    write_file_unique_name,
)
from predict_np_spot_prices.data import fetch_data, show_dfs, update_data
from predict_np_spot_prices.preprocess import preprocess


def check_args(args: argparse.Namespace):
    arg_dict = vars(args)

    command = arg_dict.get('command', None)
    data_category = arg_dict.get('data_category', None)
    start = arg_dict.get('start', None)
    end = arg_dict.get('end', None)
    area_from = arg_dict.get('area_from', None)
    area_to = arg_dict.get('area_to', None)
    special = arg_dict.get('special', None)
    dir = arg_dict.get('dir', None)
    keep = arg_dict.get('keep', None)
    remove = arg_dict.get('remove', None)
    dir_from = arg_dict.get('dir_from', None)
    dir_to = arg_dict.get('dir_to', None)
    head = arg_dict.get('head', None)
    tail = arg_dict.get('tail', None)

    if data_category is not None:
        if not DataCategory.is_data_category(data_category):
            raise ValueError(f'"{data_category}" is not a valid data category.')
    if start is not None:
        start = get_timestamp(start)
    if end is not None:
        end = get_timestamp(end)
    if area_from is not None:
        area_from = get_area_code(area_from)
    if area_to is not None:
        area_to = get_area_code(area_to)
    if keep and remove:
        raise ValueError(
            'Cannot use both --keep and --remove options simultaneously.'
        )
    if remove is None:
        keep = False

    return {
        'command': command,
        'data_category': data_category,
        'start': start,
        'end': end,
        'area_from': area_from,
        'area_to': area_to,
        'special': special,
        'dir': dir,
        'keep': keep,
        'remove': remove,
        'dir_from': dir_from,
        'dir_to': dir_to,
        'head': head,
        'tail': tail,
    }


def main() -> None:
    # make sure data dirs exist
    os.makedirs(DATA_DIR_SPECIAL, exist_ok=True)

    parser = argparse.ArgumentParser(
        prog='predict-np-spot-prices',
    )

    commands = ['fetch', 'update', 'show', 'preprocess', 'run_app', 'weather']
    subparsers = parser.add_subparsers(
        help=f'Command to execute ({", ".join(commands)}).'
    )

    def add_common_arguments(p: argparse.ArgumentParser):
        data_categories = DataCategory.list_values()
        p.add_argument(
            'data_category',
            type=str,
            nargs='?',
            choices=data_categories,
            help=f'Data category to fetch ({", ".join(data_categories)}).',
            metavar='DATA_ITEM',
        )
        p.add_argument(
            '-a',
            '--area',
            '-af',
            '--area_from',
            type=str,
            required=False,
            dest='area_from',
        )
        p.add_argument(
            '--area_to', '-at', type=str, required=False, dest='area_to'
        )
        p.add_argument('-d', '--dir', type=str, required=False)

    def add_start_and_end_arguments(p: argparse.ArgumentParser):
        p.add_argument('-s', '--start', type=str, required=False)
        p.add_argument('-e', '--end', type=str, required=False)

    def add_keep_arguments(p: argparse.ArgumentParser):
        p.add_argument(
            '-k',
            '--keep',
            action='store_true',
            help='Force keeping old files after update without asking.',
        )
        p.add_argument(
            '-r',
            '--remove',
            action='store_true',
            help='Force removal of old files after update without asking.',
        )

    # fetch
    parser_fetch = subparsers.add_parser(
        'fetch', help='Fetch data from ENTSO-E.'
    )
    parser_fetch.set_defaults(command='fetch')
    add_common_arguments(parser_fetch)
    add_start_and_end_arguments(parser_fetch)
    add_keep_arguments(parser_fetch)

    # update
    parser_update = subparsers.add_parser(
        'update', help='Update data to the current day from ENTSO-E.'
    )
    parser_update.set_defaults(command='update')
    add_common_arguments(parser_update)
    add_keep_arguments(parser_update)

    # show
    parser_show = subparsers.add_parser(
        'show', help='Show downloaded data items.'
    )
    parser_show.set_defaults(command='show')
    add_common_arguments(parser_show)
    add_start_and_end_arguments(parser_show)
    parser_show.add_argument(
        '--special',
        action='store_true',
        help='Show data from DATA_DIR_SPECIAL directory. These are dataframes '
        'preprocessed to be used for some special purposes in the streamlit '
        'app,  e.g., for demonstrating how the Norway generation data contains '
        'columns with tuple-like names, that need to be dealt with in the data '
        'cleaning process. These dataframes have been preprocessed using the '
        '--special flag.',
    )
    parser_show.add_argument(
        '-he',
        '--head',
        type=int,
        help='Number of rows to be outputted with head().',
        default=5,
    )
    parser_show.add_argument(
        '-t',
        '--tail',
        type=int,
        help='Number of rows to be outputted with tail().',
        default=5,
    )

    # preprocess
    parser_preprocess = subparsers.add_parser(
        'preprocess', help='Preprocesses the data in the data directory.'
    )
    parser_preprocess.set_defaults(command='preprocess')
    parser_preprocess.add_argument(
        '--special',
        action='store_true',
        help='Preprocess special dataframes to be used, e.g., for '
        'demonstrating some special aspects of the data cleaning process.',
    )
    parser_preprocess.add_argument(
        '-df', '--dir_from', type=str, required=False
    )
    parser_preprocess.add_argument('-dt', '--dir_to', type=str, required=False)

    # run_app
    parser_run_app = subparsers.add_parser(
        'run_app', help='Runs the Streamlit app.'
    )
    parser_run_app.set_defaults(command='run_app')

    args = parser.parse_args()

    try:
        arg_dict = check_args(args)

        if arg_dict['command'] is None:
            parser.print_help()
        elif args.command == 'fetch':
            fetch_data(
                arg_dict['data_category'],
                arg_dict['start'],
                arg_dict['end'],
                arg_dict['area_from'],
                arg_dict['area_to'],
                arg_dict['dir'],
                keep=arg_dict['keep'],
            )
        elif args.command == 'update':
            update_data(
                arg_dict['data_category'],
                arg_dict['area_from'],
                arg_dict['area_to'],
                arg_dict['dir'],
                keep=arg_dict['keep'],
            )
        elif args.command == 'show':
            if arg_dict['special']:
                arg_dict['dir'] = DATA_DIR_SPECIAL
            show_dfs(
                arg_dict['data_category'],
                arg_dict['start'],
                arg_dict['end'],
                arg_dict['area_from'],
                arg_dict['area_to'],
                arg_dict['dir'],
                arg_dict['head'],
                arg_dict['tail'],
            )
        elif args.command == 'preprocess':
            if arg_dict['special'] and arg_dict['dir_to'] is None:
                arg_dict['dir_to'] = DATA_DIR_SPECIAL
            if arg_dict['dir_to'] is None:
                arg_dict['dir_to'] = DATA_DIR_PREPROCESSED
            preprocess(
                arg_dict['special'],
                dir_from=arg_dict['dir_from'],
                dir_to=arg_dict['dir_to'],
            )
        elif args.command == 'run_app':
            import subprocess
            import signal

            proc = subprocess.Popen(
                [
                    sys.executable,
                    '-m',
                    'streamlit',
                    'run',
                    'src/predict_np_spot_prices/Home.py',
                ],
                preexec_fn=None if sys.platform == 'win32' else os.setsid,
            )

            try:
                proc.wait()
            except KeyboardInterrupt:
                print(
                    '\nKeyboardInterrupt detected. Shutting down Streamlit...'
                )

                if sys.platform == 'win32':
                    proc.terminate()  # sends CTRL-BREAK equivalent on Windows
                else:
                    # kill process group on Unix
                    os.killpg(proc.pid, signal.SIGTERM)

                proc.wait()  # ensure the process has fully terminated
                print('Streamlit stopped.')

    except KeyboardInterrupt:
        print('\nKeyboardInterrupt detected. Exiting...')
        sys.exit(1)
    except Exception as e:
        error_message = traceback.format_exc()
        error_path = write_file_unique_name(
            'error.log',
            bytearray(error_message, 'utf-8'),
            silent=True,
        )
        print(
            f'Something went wrong: {e}; stack trace has been written to "{error_path}".',
            file=sys.stderr,
        )
        sys.exit(1)

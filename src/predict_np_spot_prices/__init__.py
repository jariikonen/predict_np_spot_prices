import argparse
import os
import sys
import traceback
from predict_np_spot_prices.common import (
    DATA_DIR,
    DATA_DIR_EDA,
    DataItem,
    get_area_code,
    get_timestamp,
    write_file_unique_name,
)
from predict_np_spot_prices.data import fetch_data, show_dfs, update_data
from predict_np_spot_prices.preprocess import preprocess


def check_args(args: argparse.Namespace):
    arg_dict = vars(args)
    print(arg_dict)

    command = arg_dict.get('command', None)
    data_item = arg_dict.get('data_item', None)
    start = arg_dict.get('start', None)
    end = arg_dict.get('end', None)
    area_from = arg_dict.get('area_from', None)
    area_to = arg_dict.get('area_to', None)
    eda = arg_dict.get('eda', None)
    dir = arg_dict.get('dir', None)

    if data_item is not None:
        if not DataItem.is_data_item(data_item):
            raise ValueError(f'"{data_item}" is not a valid data item.')
    if start is not None:
        start = get_timestamp(start)
    if end is not None:
        end = get_timestamp(end)
    if area_from is not None:
        area_from = get_area_code(area_from)
    if area_to is not None:
        area_to = get_area_code(area_to)
    if dir is None:
        dir = DATA_DIR

    return {
        'command': command,
        'data_item': data_item,
        'start': start,
        'end': end,
        'area_from': area_from,
        'area_to': area_to,
        'eda': eda,
        'dir': dir,
    }


def main() -> None:
    # make sure data dirs exist
    os.makedirs(DATA_DIR_EDA, exist_ok=True)

    parser = argparse.ArgumentParser(
        prog='predict-np-spot-prices',
    )

    commands = ['fetch', 'update', 'show', 'preprocess', 'run_app']
    subparsers = parser.add_subparsers(
        help=f'Command to execute ({", ".join(commands)}).'
    )

    def add_common_arguments(p: argparse.ArgumentParser):
        data_items = DataItem.list_values()
        p.add_argument(
            'data_item',
            type=str,
            nargs='?',
            choices=data_items,
            help=f'Data item to fetch ({", ".join(data_items)}).',
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
            '-t', '--area_to', '-at', type=str, required=False, dest='area_to'
        )
        p.add_argument('-d', '--dir', type=str, required=False)

    def add_start_and_end_arguments(p: argparse.ArgumentParser):
        p.add_argument('-s', '--start', type=str, required=False)
        p.add_argument('-e', '--end', type=str, required=False)

    # fetch
    parser_fetch = subparsers.add_parser(
        'fetch', help='Fetch data from ENTSO-E.'
    )
    parser_fetch.set_defaults(command='fetch')
    add_common_arguments(parser_fetch)
    add_start_and_end_arguments(parser_fetch)

    # update
    parser_update = subparsers.add_parser(
        'update', help='Update data to the current day from ENTSO-E.'
    )
    parser_update.set_defaults(command='update')
    add_common_arguments(parser_update)

    # show
    parser_show = subparsers.add_parser(
        'show', help='Show downloaded data items.'
    )
    parser_show.set_defaults(command='show')
    add_common_arguments(parser_show)
    add_start_and_end_arguments(parser_show)
    parser_show.add_argument(
        '--eda',
        action='store_true',
        help='Show data from DATA_DIR_EDA directory. These are the dataframes '
        'preprocessed for exploratory data analysis (using --eda flag).',
    )

    # preprocess
    parser_preprocess = subparsers.add_parser(
        'preprocess', help='Preprocesses the data in the data directory.'
    )
    parser_preprocess.set_defaults(command='preprocess')
    parser_preprocess.add_argument(
        '--eda',
        action='store_true',
        help='Preprocess data for exploratory data analysis.',
    )

    # run_app
    parser_run_app = subparsers.add_parser(
        'run_app', help='Runs the Streamlit app.'
    )
    parser_run_app.set_defaults(command='run_app')

    args = parser.parse_args()

    try:
        arg_dict = check_args(args)
        print(arg_dict)

        if arg_dict['command'] is None:
            parser.print_help()
        elif args.command == 'fetch':
            fetch_data(
                arg_dict['data_item'],
                arg_dict['start'],
                arg_dict['end'],
                arg_dict['area_from'],
                arg_dict['area_to'],
                arg_dict['dir'],
            )
        elif args.command == 'update':
            update_data(
                arg_dict['data_item'],
                arg_dict['area_from'],
                arg_dict['area_to'],
                arg_dict['dir'],
            )
        elif args.command == 'show':
            show_dfs(
                arg_dict['data_item'],
                arg_dict['start'],
                arg_dict['end'],
                arg_dict['area_from'],
                arg_dict['area_to'],
                arg_dict['eda'],
            )
        elif args.command == 'preprocess':
            preprocess(arg_dict['eda'])
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

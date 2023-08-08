import argparse
FILES = [
    'test_dict.pk',
]

def WorkerArgsBuilder():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="source name",
                        type=str)
    parser.add_argument("-k", "--opt_key", help="opt_key",
                        type=str)
    parser.add_argument("--file_name_suffix", help="file_name_suffix",
                        default="", type=str)
    parser.add_argument("-p", "--perturbation", help="folder name",
                        action="store_true")
    parser.add_argument("--file", choices=FILES,
                        help='data file to evaluate on')
    parser.add_argument("--index", default=None, type=int, help="index of data")
   
    args = parser.parse_args()
    return args

def WorkerCalculatorsArgsBuilder():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="Name of the data source.",
                        type=str)
    parser.add_argument("--method", help="source name",
                        type=str)
    parser.add_argument("--bias", help="source name",
                        type=str)
    parser.add_argument("-k", "--opt_key", help="opt_key",
                        type=str)
    parser.add_argument("--file_name_suffix", help="file_name_suffix",
                        default="", type=str)
    parser.add_argument("-p", "--perturbation", help="folder name",
                        action="store_true")
    parser.add_argument("--file", choices=FILES,
                        help='data file to evaluate on')
    parser.add_argument("--index", default=None, type=int, help="index of data")
   
    args = parser.parse_args()
    return args

def WorkerComplexArgsBuilder():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="source name",
                        type=str)                   
    parser.add_argument("--version", help="version",
                        type=str)
    parser.add_argument("--seed", help="seed",
                        type=str)
    parser.add_argument("--standard_lesson", help="standard_lesson",
                        type=str)

    parser.add_argument("--file_name_suffix", help="file_name_suffix",
                        default="", type=str)
    parser.add_argument("-p", "--perturbation", help="folder name",
                        action="store_true")
    parser.add_argument("--file", choices=FILES,
                        help='data file to evaluate on')
    parser.add_argument("--index", default=None, type=int, help="index of data")
   
    args = parser.parse_args()
    return args
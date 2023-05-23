import argparse

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--M',default=8)
    parser.add_argument('--N',default=16)
    parser.add_argument('--K',default=4)

    return parser.parse_args()

    
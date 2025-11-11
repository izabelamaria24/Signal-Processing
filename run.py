import argparse
import sys

from Lab1 import run as run_lab1
from Lab2 import run as run_lab2
from Lab3 import run as run_lab3
from Lab4 import run as run_lab4
from Lab5 import run as run_lab5
from Lab6 import run as run_lab6


def run_one(lab_number: int):
    if lab_number == 1:
        run_lab1()
    elif lab_number == 2:
        run_lab2()
    elif lab_number == 3:
        run_lab3()
    elif lab_number == 4:
        run_lab4()
    elif lab_number == 5:
        run_lab5()
    elif lab_number == 6:
        run_lab6()
    else:
        raise ValueError("Lab number must be between 1 and 5.")


def run_all():
    for i in range(1, 7):
        print(f"\n=== Running Lab {i} ===\n")
        run_one(i)


def main():
    parser = argparse.ArgumentParser(description="Run Signal Processing Labs")
    parser.add_argument("--lab", type=str, default="all",
                        help="Lab to run: '1'..'6' or 'all'")
    args = parser.parse_args()
    if args.lab == "all":
        run_all()
    else:
        try:
            num = int(args.lab)
        except ValueError:
            print("Invalid --lab argument. Use 'all' or a number 1..6.")
            sys.exit(1)
        run_one(num)


if __name__ == "__main__":
    main()



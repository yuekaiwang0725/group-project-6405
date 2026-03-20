import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="EE6405 sentiment project entry point")
    parser.add_argument(
        "command",
        choices=["prepare-data"],
        help="Project command to execute",
    )
    args = parser.parse_args()

    if args.command == "prepare-data":
        print("TODO: implement dataset preparation pipeline.")


if __name__ == "__main__":
    main()

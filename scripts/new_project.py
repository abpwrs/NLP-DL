import argparse
import os
import json


def main(argv):
    for directory in ["data", "model", "out", "src", "job"]:
        if not os.path.isdir(os.path.join(os.getcwd(), directory, argv.name)):
            os.mkdir(os.path.join(os.getcwd(), directory, argv.name))

    config = {"BASE_CONFIG": str(os.path.join(os.getcwd(), "config.json"))}

    with open(os.path.join(os.getcwd(), "src", argv.name, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create a new project")
    parser.add_argument("-n", "--name", required=True, help="name of the new project")
    main(parser.parse_args())

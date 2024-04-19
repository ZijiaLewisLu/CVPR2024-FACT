import os

def get_project_base():
    src_dir = os.path.dirname(os.path.realpath(__file__))
    base = os.path.dirname(src_dir) + "/"
    return base


if __name__ == "__main__":
    print(get_project_base())

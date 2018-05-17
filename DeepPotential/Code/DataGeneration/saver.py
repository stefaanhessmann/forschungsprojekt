import os


def create_path(path):
    part_path = path.split('/')
    new_path = ''
    for folder in part_path:
        new_path = os.path.join(new_path, folder)
        if not os.path.exists(new_path):
            os.mkdir(new_path)

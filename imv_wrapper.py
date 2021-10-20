from pathlib import Path
import socket
from subprocess import Popen, PIPE
import shutil
import time


class ImvWindow:
    def __init__(self, proc, images: list):
        self.proc = proc
        self.images = images

    @classmethod
    def view_images(cls, images):
        image_paths = [str(i) for i in images]
        win = Popen(['imv'] + image_paths, stdout=PIPE, text=True)
        return cls(win, images).poll()

    def send_command(self, pid: int, cmd: str, attempts: int = 1) -> None:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock_file = f'/run/user/1000/imv-{pid}.sock'
        try:
            client.connect(sock_file)
        except FileNotFoundError:
            time.sleep(0.1)
            if attempts < 3:
                attempts += 1
                self.send_command(pid, cmd, attempts)
            else:
                raise Exception(f'Too many attempts to send {cmd} to {pid}')
        client.send(cmd.encode())

    def poll(self, **kwargs):
        print(kwargs)

        while self.proc.poll() is None:
            assert self.proc.stdout
            line = self.proc.stdout.readline()
            index = 0
            action = None
            try:
                index = int(line) - 1  # imv starts counting at 1 not 0
                image = self.images[index]
            except ValueError:
                try:
                    line = line.split(' ')
                    action, index = line[0], int(line[1]) - 1
                except IndexError:
                    action = line

            if action == 'tag':
                dest = Path('data/pos') / image.name
                shutil.move(image, dest)

            elif action == 'untag':
                dest = Path('data/neg') / image.name
                shutil.move(image, dest)

            else:
                time.sleep(.1)
                continue
        self.cleanup()

    def close_image(self, index: int):
        self.send_command(self.proc.pid, f'close {index + 1}')
        with open(f'/tmp/{self.proc.pid}-imv-overlay.txt', 'r+') as f:
            new = []
            for line in f.readlines():
                split_line = line.split(' ', 1)
                idx, text = int(split_line[0]), split_line[1]
                if idx == 1:
                    return
                elif idx == index + 1:
                    continue
                new.append(f'{idx - 1} {text}\n')
            f.writelines(new)

    def cleanup(self):
        print('CLEAN', self.proc.pid)
        sock_file = Path(f'/run/user/1000/imv-{self.proc.pid}.sock')
        overlay_file = Path(f'/tmp/{self.proc.pid}-imv-overlay.txt')
        if sock_file.exists():
            sock_file.unlink()
        if overlay_file.exists():
            overlay_file.unlink()

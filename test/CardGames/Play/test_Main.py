import io
import os
import shutil
import socket
import subprocess
import threading
from contextlib import redirect_stderr, redirect_stdout
from unittest import TestCase, skipUnless

from src.CardGames.Play.Server import make_server
from src.CardGames.Play.__main__ import already_serving, main

__author__ = 'frank.ma'


class TestBusyPort(TestCase):
    def test_reuses_a_running_card_games_server(self):
        server = make_server(port=0)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            self.assertTrue(already_serving(port))
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(['blackjack', '--port', str(port), '--no-browser'])
            self.assertEqual(code, 0)
            self.assertIn('already running at http://127.0.0.1:%i/blackjack' % port, out.getvalue())
        finally:
            server.shutdown()
            server.server_close()

    def test_another_program_on_the_port(self):
        # a plain socket that listens but never answers like the card games server
        other = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        other.bind(('127.0.0.1', 0))
        other.listen(1)
        port = other.getsockname()[1]
        try:
            self.assertFalse(already_serving(port))
            err = io.StringIO()
            with redirect_stderr(err):
                code = main(['--port', str(port), '--no-browser'])
            self.assertEqual(code, 1)
            self.assertIn('Port %i is in use by another program' % port, err.getvalue())
            self.assertIn('--port %i' % (port + 1), err.getvalue())
        finally:
            other.close()


LAUNCHER = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'viz', 'Play Card Games.command')


@skipUnless(shutil.which('bash'), 'the launcher is a bash script')
class TestLauncher(TestCase):
    def test_script(self):
        self.assertTrue(os.access(LAUNCHER, os.X_OK), 'the launcher must be executable to double-click.')
        self.assertEqual(subprocess.run(['bash', '-n', LAUNCHER]).returncode, 0)

    def test_opens_a_running_server(self):
        server = make_server(port=0)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            done = subprocess.run([LAUNCHER, '--port', str(port), '--no-browser'], capture_output=True, text=True,
                                  timeout=60)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertIn('already running at http://127.0.0.1:%i/' % port, done.stdout)
        finally:
            server.shutdown()
            server.server_close()

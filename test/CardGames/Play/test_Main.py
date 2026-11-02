import io
import os
import shutil
import socket
import subprocess
import threading
import urllib.error
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from unittest import TestCase, skipUnless

from src.CardGames.Play.Server import SHUTDOWN_HEADER, build_id, make_server
from src.CardGames.Play.__main__ import claim, main, probe

__author__ = 'frank.ma'


def serve(build=None):
    # a card games server on a free port, in the background; build '' stands for one from before builds were reported
    server = make_server(port=0, build=build)

    def run():
        # as main does: serve until stopped, then free the port
        server.serve_forever()
        server.server_close()
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return server, thread


def stop(server):
    server.shutdown()
    server.server_close()


class TestBusyPort(TestCase):
    def test_reuses_a_server_on_this_code(self):
        server, _ = serve()
        port = server.server_address[1]
        try:
            self.assertEqual(probe(port), build_id())
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(['blackjack', '--port', str(port), '--no-browser'])
            self.assertEqual(code, 0)
            self.assertIn('already running at http://127.0.0.1:%i/blackjack' % port, out.getvalue())
        finally:
            stop(server)

    def test_replaces_a_server_on_other_code(self):
        old, thread = serve(build='0ld')
        port = old.server_address[1]
        new, how = claim(port)
        try:
            self.assertEqual(how, 'replaced')
            thread.join(5)
            self.assertFalse(thread.is_alive(), 'the old server stopped serving.')
            self.assertTrue(old.replaced)
            self.assertEqual(new.server_address[1], port)
            threading.Thread(target=new.serve_forever, daemon=True).start()
            self.assertEqual(probe(port), build_id(), 'this code now answers on the port.')
        finally:
            old.server_close()
            stop(new)

    def test_a_server_too_old_to_ask(self):
        old, _ = serve(build='')
        port = old.server_address[1]
        try:
            self.assertEqual(probe(port), '')
            err = io.StringIO()
            with redirect_stderr(err):
                code = main(['--port', str(port), '--no-browser'])
            self.assertEqual(code, 1)
            self.assertIn('An older card games server is running on port %i' % port, err.getvalue())
            self.assertEqual(probe(port), '', 'left running: only you can stop it.')
        finally:
            stop(old)

    def test_shutdown_needs_the_build_header(self):
        server, thread = serve(build='abc')
        port = server.server_address[1]
        try:
            for headers in ({}, {SHUTDOWN_HEADER: 'xyz'}):
                request = urllib.request.Request('http://127.0.0.1:%i/api/shutdown' % port, data=b'', method='POST',
                                                 headers=headers)
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(request, timeout=2)
                self.assertEqual(error.exception.code, 403)
            self.assertEqual(probe(port), 'abc', 'still serving.')
            # a page on another site cannot even ask: the browser's preflight for the custom header goes unanswered
            preflight = urllib.request.Request('http://127.0.0.1:%i/api/shutdown' % port, method='OPTIONS')
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(preflight, timeout=2)
            self.assertNotIn('Access-Control-Allow-Headers', error.exception.headers)
        finally:
            stop(server)
        self.assertFalse(server.replaced)

    def test_another_program_on_the_port(self):
        # a plain socket that listens but never answers like the card games server
        other = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        other.bind(('127.0.0.1', 0))
        other.listen(1)
        port = other.getsockname()[1]
        try:
            self.assertIsNone(probe(port))
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
        server, _ = serve()
        port = server.server_address[1]
        try:
            done = subprocess.run([LAUNCHER, '--port', str(port), '--no-browser'], capture_output=True, text=True,
                                  timeout=60)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertIn('already running at http://127.0.0.1:%i/' % port, done.stdout)
        finally:
            stop(server)

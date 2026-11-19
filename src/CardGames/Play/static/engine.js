// the web version's game server: a worker that runs the play server's own Python (src/CardGames/Play/Api.py) with
// Pyodide, so the pages ask it what they would ask the local server. Only used on the static site, e.g. GitHub Pages
const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.27.2/full/';
importScripts(PYODIDE + 'pyodide.js');

let call = null;
const ready = (async () => {
  const saved = await new Promise(resolve => { self.onmessage = e => resolve(e.data.init); });
  self.postMessage({ loading: 'Loading the game engine' });
  const py = await loadPyodide({ indexURL: PYODIDE });
  await py.loadPackage('numpy');
  // the card games' Python, packed by the site builder next to this file
  const zip = await fetch('cardgames.zip');
  if (!zip.ok) throw new Error('could not load the game code (' + zip.status + ').');
  py.unpackArchive(await zip.arrayBuffer(), 'zip', { extractDir: '/home/pyodide/cardgames' });
  py.globals.set('saved_progress', saved === undefined ? null : saved);
  // progress is saved by the page, in the browser's storage, as the local server saves it to a file
  py.globals.set('post_save', text => self.postMessage({ save: text }));
  py.runPython(`
import json, sys
sys.path.insert(0, '/home/pyodide/cardgames')
from src.CardGames.Play.Api import MemoryStore, Sessions, handle
sessions = Sessions(store=MemoryStore(saved_progress, post_save))

def call(method, path, body):
    status, payload = handle(sessions, method, path, json.loads(body) if body else None)
    return json.dumps([status, payload])
`);
  call = py.globals.get('call');
  self.postMessage({ loading: null });
})();

ready.catch(e => self.postMessage({ loading: null, failed: String(e && e.message || e) }));

// requests wait until the engine is up, then run one at a time, as each answers before the next is read
let queue = Promise.resolve();
function answer({ id, method, path, body }) {
  queue = queue.then(async () => {
    try {
      await ready;
      const [status, payload] = JSON.parse(call(method, path, body === undefined ? null : JSON.stringify(body)));
      self.postMessage({ id, status, payload });
    } catch (e) {
      self.postMessage({ id, status: 503, payload: { error: 'The game engine could not run: ' + (e && e.message || e) } });
    }
  });
}
self.addEventListener('message', e => { if (e.data && e.data.id !== undefined) answer(e.data); });

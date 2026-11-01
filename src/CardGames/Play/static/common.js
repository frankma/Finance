// shared by the lobby and every table page: requests to the play server, cards, small formatting helpers
const $ = id => document.getElementById(id);
const SUIT = { S: ['♠', false], H: ['♥', true], D: ['♦', true], C: ['♣', false] };
const RANK = { T: '10' };

async function api(method, path, body) {
  const res = await fetch(path, { method, headers: { 'Content-Type': 'application/json' },
                                  body: body === undefined ? undefined : JSON.stringify(body) });
  const data = await res.json().catch(() => ({ error: 'bad response' }));
  if (!res.ok) { const e = new Error(data.error || res.statusText); e.status = res.status; throw e; }
  return data;
}

// one card from its short form ('AS', 'TH', 'BJ'), or a face-down back
function cardEl(code) {
  const el = document.createElement('div');
  if (!code) { el.className = 'card back'; el.setAttribute('aria-label', 'face-down card'); return el; }
  if (code === 'BJ' || code === 'RJ') {
    el.className = 'card' + (code === 'RJ' ? ' red' : '');
    el.innerHTML = `<span>J</span><span class="s">★</span>`;
    el.setAttribute('aria-label', code === 'RJ' ? 'red joker' : 'black joker');
    return el;
  }
  const rank = code.slice(0, -1), [sym, red] = SUIT[code.slice(-1)];
  el.className = 'card' + (red ? ' red' : '');
  el.innerHTML = `<span>${RANK[rank] || rank}</span><span class="s">${sym}</span>`;
  el.setAttribute('aria-label', (RANK[rank] || rank) + ' of ' + { S: 'spades', H: 'hearts', D: 'diamonds', C: 'clubs' }[code.slice(-1)]);
  return el;
}
function drawCards(host, codes) { host.replaceChildren(...codes.map(cardEl)); }

const fmt = (x, d = 0) => Number(x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
const signed = (x, d = 0) => (x > 0 ? '+' : x < 0 ? '−' : '±') + fmt(Math.abs(x), d);

// remember a per-viewer preference; storage may be unavailable, so never depend on it
const prefs = {
  get(key, fallback) { try { const v = localStorage.getItem('cg-' + key); return v === null ? fallback : JSON.parse(v); } catch (e) { return fallback; } },
  set(key, value) { try { localStorage.setItem('cg-' + key, JSON.stringify(value)); } catch (e) {} },
};

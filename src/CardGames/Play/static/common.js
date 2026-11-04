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
    // named on the face, not by colour alone, and not a bare J that reads as a jack
    const name = code === 'RJ' ? 'red joker' : 'black joker';
    el.className = 'card joker' + (code === 'RJ' ? ' red' : '');
    el.innerHTML = `<span>${code}</span><span class="s">★</span>`;
    el.setAttribute('aria-label', name); el.title = name[0].toUpperCase() + name.slice(1);
    return el;
  }
  const rank = code.slice(0, -1), [sym, red] = SUIT[code.slice(-1)];
  el.className = 'card' + (red ? ' red' : '');
  el.innerHTML = `<span>${RANK[rank] || rank}</span><span class="s">${sym}</span>`;
  el.setAttribute('aria-label', (RANK[rank] || rank) + ' of ' + { S: 'spades', H: 'hearts', D: 'diamonds', C: 'clubs' }[code.slice(-1)]);
  return el;
}
function drawCards(host, codes) { host.replaceChildren(...codes.map(cardEl)); }

// the error line on a table page; pass null to clear it
function showError(e) { const el = $('err'); el.hidden = !e; el.textContent = e ? e.message || String(e) : ''; }
const cap = s => s[0].toUpperCase() + s.slice(1);

// your seat at one game on the play server, kept across reloads by its session id in this browser; mode 'learn'
// keeps the game's lessons apart from free play
function seat(game, mode) {
  let sid = null;
  const key = game + (mode ? '-' + mode : '') + '-session';
  return {
    // resume this browser's session, or sit down at a new table with these options
    async open(fresh, options) {
      sid = fresh ? null : prefs.get(key, null);
      if (sid) { try { return await api('GET', `/api/${game}/${sid}`); } catch (e) { sid = null; } }
      const made = await api('POST', `/api/${game}/new`, options || {});
      sid = made.id; prefs.set(key, sid);
      return made.view;
    },
    act: action => api('POST', `/api/${game}/${sid}/act`, action),
    advice: () => api('GET', `/api/${game}/${sid}/advice`),
  };
}

const fmt = (x, d = 0) => Number(x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
const signed = (x, d = 0) => (x > 0 ? '+' : x < 0 ? '−' : '±') + fmt(Math.abs(x), d);

// remember a per-viewer preference; storage may be unavailable, so never depend on it
const prefs = {
  get(key, fallback) { try { const v = localStorage.getItem('cg-' + key); return v === null ? fallback : JSON.parse(v); } catch (e) { return fallback; } },
  set(key, value) { try { localStorage.setItem('cg-' + key, JSON.stringify(value)); } catch (e) {} },
};

// ---- how to play: one rules text per game (static/rules-<game>.html), shown as its own page and in a table's panel ----

// examples in a rules text: <span data-cards="AS KD"></span> draws those cards, '_' a face-down one
function drawExamples(root) {
  root.querySelectorAll('[data-cards]').forEach(el => {
    el.classList.add('cards', 'eg');
    drawCards(el, el.dataset.cards.split(' ').map(c => c === '_' ? null : c));
  });
}

const rulesCache = {};
async function loadRules(game) {
  if (!rulesCache[game]) {
    const res = await fetch(`/static/rules-${game}.html`);
    if (!res.ok) throw new Error('Could not load the rules.');
    rulesCache[game] = await res.text();
  }
  return rulesCache[game];
}

// a panel beside the table holding the game's rules, opened by the ? key, a "Rules" link or any [data-rules] link;
// note() gives a line about this table's own settings, shown above the text
function rulesPanel(game, note) {
  const panel = document.createElement('aside');
  panel.className = 'rules-panel'; panel.hidden = true; panel.setAttribute('aria-label', 'How to play');
  panel.innerHTML = `<div class="rules-head"><span class="kicker">How to play</span>
    <a href="/${game}/rules" target="_blank" rel="noopener">Open as a page</a>
    <button type="button" class="rules-close">Close<kbd>Esc</kbd></button></div>
    <p class="this-table small" hidden></p><div class="rules"></div>`;
  document.body.append(panel);
  const body = panel.querySelector('.rules');
  const close = () => { panel.hidden = true; };
  async function open(section) {
    if (!body.childElementCount) { body.innerHTML = await loadRules(game); drawExamples(body); }
    const line = panel.querySelector('.this-table'), text = note ? note() : '';
    line.hidden = !text; line.textContent = text ? 'This table: ' + text : '';
    panel.hidden = false;
    // scroll the panel only, never the table behind it
    const target = section && body.querySelector('#' + section);
    body.scrollTop = target ? target.offsetTop - body.offsetTop - 8 : 0;
  }
  panel.querySelector('.rules-close').onclick = close;
  document.addEventListener('click', e => {
    const link = e.target.closest('[data-rules]');
    if (link) { e.preventDefault(); open(link.dataset.rules).catch(showError); }
  });
  // capture first, so Esc and ? reach the panel before the table's own keys
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !panel.hidden) { e.stopPropagation(); close(); }
    else if (e.key === '?' && e.target.tagName !== 'INPUT') {
      // the same section the Rules link would open, which the table keeps up to date
      e.preventDefault(); e.stopPropagation();
      const link = document.getElementById('rulesLink');
      panel.hidden ? open(link ? link.dataset.rules : '').catch(showError) : close();
    }
  }, true);
  return { open, close, get isOpen() { return !panel.hidden; } };
}

// put links to the rules on the terms a text uses, e.g. the coach's reason; terms maps a phrase to a section id. All
// the phrases are matched together, longest first, so 'straight flush' is never read as 'straight', and each section is
// linked once in a text however many of its phrases appear
function linkTerms(text, terms) {
  const escape = s => s.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  const phrases = Object.keys(terms).sort((a, b) => b.length - a.length);
  if (!phrases.length) return escape(text);
  const sections = Object.fromEntries(phrases.map(p => [p.toLowerCase(), terms[p]]));
  const alternatives = phrases.map(p => p.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
  const linked = new Set();
  return escape(text).replace(new RegExp(`(^|[^\\w-])(${alternatives})(?![\\w-])`, 'gi'), (match, before, phrase) => {
    const section = sections[phrase.toLowerCase()];
    if (linked.has(section)) return match;
    linked.add(section);
    return `${before}<a href="#" data-rules="${section}">${phrase}</a>`;
  });
}

// ---- guided practice: the lesson box above a table in learn mode ----

// 'learn' when the page was opened with ?mode=learn, else '' for free play
const MODE = new URLSearchParams(location.search).get('mode') === 'learn' ? 'learn' : '';

// draw the lesson in front of you into host; act sends the page's actions, terms links words to the rules
function renderLesson(host, lesson, act, terms) {
  host.hidden = !lesson;
  if (!lesson) return;
  const feedback = lesson.feedback;
  const steps = lesson.n_steps === 1 ? '' : ` · step ${Math.min(lesson.step + 1, lesson.n_steps)} of ${lesson.n_steps}`;
  host.innerHTML = `
    <div class="lesson-head"><span class="kicker">Lesson ${lesson.index + 1} of ${lesson.lessons.length}${steps}</span>
      <a href="#" data-rules="${lesson.tags[0] || ''}">Rules</a></div>
    <h2>${lesson.title}</h2>
    ${feedback ? `<p class="lesson-feedback ${feedback.kind}">${feedback.kind === 'right'
      ? linkTerms(feedback.text, terms).replace(/^Right([:.])/, '<b>Right$1</b>')
      : feedback.kind === 'hint' ? '<b>Not quite.</b> ' + linkTerms(feedback.text, terms) : linkTerms(feedback.text, terms)}</p>` : ''}
    ${lesson.prompt ? `<p class="lesson-prompt">${linkTerms(lesson.prompt, terms)}</p>` : ''}
    ${lesson.complete ? `<div class="row"><button type="button" class="primary" data-lesson="${Math.min(lesson.index + 1, lesson.lessons.length - 1)}" ${lesson.index + 1 >= lesson.lessons.length ? 'hidden' : ''}>Next lesson</button>
      <button type="button" data-lesson="${lesson.index}">Try it again</button>
      ${lesson.index + 1 >= lesson.lessons.length ? '<span class="small">That was the last lesson. <a href="?">Play freely</a> with the coach, or pick any lesson again.</span>' : ''}</div>` : ''}
    <ol class="lesson-list">${lesson.lessons.map((x, k) => `<li class="${k === lesson.index ? 'current' : ''}">
      <a href="#" data-lesson="${k}">${x.title}</a>${x.done ? ' <span class="tick" aria-label="done">✓</span>' : ''}</li>`).join('')}</ol>`;
  host.querySelectorAll('[data-lesson]').forEach(el => el.onclick = e => { e.preventDefault(); act({ type: 'lesson', index: +el.dataset.lesson }); });
}

'use strict';

const benchmarks = [
  {name:'OVOBench', metric:'Overall', ours:72.1, base:58.8, digits:1},
  {name:'StreamingBench', metric:'Real-Time', ours:86.9, base:81.8, digits:1},
  {name:'OVBench', metric:'Average', ours:66.8, base:55.4, digits:1},
  {name:'ODVBench', metric:'Overall', ours:71.3, base:57.6, digits:1},
  {name:'ProactiveVQA', metric:'Average', ours:48.7, base:34.3, digits:1},
  {name:'OmniMMI', metric:'Average', ours:36.6, base:29.4, digits:1},
  {name:'OVO-Timing', metric:'Average F1', ours:41.6, base:29.4, digits:1},
  {name:'ViSpeak', metric:'Average', ours:2.87, base:2.41, digits:2}
];
const models = [
  {name:'VideoLLM-Online',size:'7B',scores:[12.8,36.0,9.6,null,23.6,null,6.9,null]},
  {name:'Flash-Vstream',size:'7B',scores:[33.2,23.2,31.2,35.7,null,null,null,null]},
  {name:'Dispider',size:'7B + 1.5B',scores:[41.8,67.6,null,45.2,null,null,null,null]},
  {name:'VideoChat-Online',size:'4B',scores:[null,null,54.9,54.5,null,null,null,null]},
  {name:'TimeChat-Online',size:'7B',scores:[47.6,75.4,null,null,null,null,null,null]},
  {name:'StreamBridge',size:'7B + 0.5B',scores:[62.6,77.0,null,null,null,null,null,null]},
  {name:'StreamForest',size:'7B',scores:[55.6,77.3,65.4,59.9,null,null,null,null]},
  {name:'StreamingVLM',size:'7B',scores:[null,null,null,null,17.9,null,null,null]},
  {name:'MMDuet-2',size:'3B',scores:[null,null,null,null,39.8,null,20.5,null]},
  {name:'Streamo',size:'7B',scores:[57.9,null,null,null,null,null,null,null]},
  {name:'Em-Garde',size:'7B + 2B',scores:[null,null,null,null,null,null,31.0,null]},
  {name:'VideoChat3',size:'4B',scores:[58.5,81.9,62.5,70.8,37.6,24.6,33.6,1.05]},
  {name:'Mage-VL',size:'4B',scores:[58.5,82.1,57.5,64.1,25.6,15.6,21.7,.85]},
  {name:'JoyAI-VL-Interaction',size:'8B',scores:[59.1,82.7,62.3,68.6,29.6,17.8,20.0,2.16]},
  {name:'AURA',size:'8B',scores:[65.3,83.2,58.3,58.8,30.8,25.4,11.1,.87]},
  {name:'MOSS-VL-Realtime',size:'11B',scores:[70.2,82.9,53.7,63.9,47.2,32.7,38.5,2.48]},
  {name:'Qwen3-VL (base)',size:'4B',scores:[58.8,81.8,55.4,57.6,34.3,29.4,29.4,2.41]},
  {name:'OneStreamer',size:'4B',scores:[72.1,86.9,66.8,71.3,48.7,36.6,41.6,2.87]}
];

const examples = {
  memory: {
    times:['00:06','00:14','00:29','00:42','00:59','01:29','01:47'],
    question:'Where can my kids go to read?',
    questionTime:'01:47',
    answer:'The small yellow table upstairs.',
    captions:[
      'An early detail: books on a small yellow table.',
      'The camera moves away from the reading corner.',
      'Local details are recorded as the stream unfolds.',
      'The scene changes. Earlier records remain available.',
      'Recent frames show the staircase and artwork.',
      'A broader summary preserves the scene context.',
      'The reading corner is no longer in view.'
    ],
    alts:[
      'A small yellow table beneath colorful bookshelves and a Reading is hoot sign.',
      'A white landing and black wire wall sculpture beside a hallway.',
      'The white landing and wall sculpture seen from farther along the hallway.',
      'A white stair handrail beside a colorful abstract painting.',
      'The abstract painting on the stairwell wall.',
      'A view from the staircase toward the upstairs hallway.',
      'The camera faces a living room from the hallway; the yellow reading table is no longer visible.'
    ],
    records:[
      {kind:'Observe',text:'… A yellow table with a white paper surface holds a blue “Hoot” book, a small book and a blue can …'},
      {kind:'Observe',text:'… White raised platform, black wall sculpture, hallway and bedroom doorway.'},
      {kind:'Observe',text:'… Reading nook “Reading is hoot!” and bookshelves are visible. …'},
      {kind:'Observe',text:'… A white staircase handrail and a colorful abstract painting appear …'},
      {kind:'Observe',text:'… Colorful abstract painting beside the handrail. …'},
      {kind:'Summary',text:'… Top of the staircase and a hallway.'}
    ]
  },
  counting: {
    times:['00:00','00:04','00:05','00:09','00:11','00:15','00:17'],
    question:'How many times does the event: showing something to the camera happen?',
    questionTime:'00:00',
    outputs:{2:'1 time',4:'2 times',6:'3 times'},
    captions:[
      'The instruction arrives before the target events.',
      'Keep observing. No new showing event yet.',
      'A fork is shown to the camera. Count: one.',
      'Remain silent between target events.',
      'A pencil is shown to the camera. Count: two.',
      'Wait for new visual evidence.',
      'A pen is shown to the camera. Count: three.'
    ],
    alts:[
      'A person sits at a table beside a mug patterned with blue hearts.',
      'The person sits behind the table without displaying a new object.',
      'The person holds a fork up toward the camera.',
      'The person has lowered the fork and is not displaying a new object.',
      'The person holds a yellow pencil up toward the camera.',
      'The person waits behind the table after displaying the pencil.',
      'The person holds a black pen up toward the camera.'
    ]
  }
};

const byId = id => document.getElementById(id);
const escapeHTML = text => String(text).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
let activeExample = 'memory';
let frameIndex = 6;
let replayTimer = null;

function recordMarkup(record, time, extraClass = '') {
  return `<div class="memory-record ${record.kind === 'Summary' ? 'summary-record' : ''} ${extraClass}"><div class="record-meta"><span>${escapeHTML(record.kind)}</span><time>${time}</time></div><p>${escapeHTML(record.text)}</p></div>`;
}

function buildFrameStrip() {
  const example = examples[activeExample];
  byId('frame-strip').innerHTML = example.times.map((time, index) => `<button class="frame-thumb ${activeExample === 'memory' && index === 0 ? 'evidence-thumb' : ''}" aria-label="View keyframe ${index + 1}, ${time}" aria-pressed="${index === frameIndex}" data-frame="${index}"><img src="assets/frames/${activeExample}-${index}-thumb.webp" width="240" height="135" alt=""><time>${time}</time></button>`).join('');
}

function renderFrame() {
  const example = examples[activeExample];
  const currentTime = example.times[frameIndex];
  byId('demo-frame').src = `assets/frames/${activeExample}-${frameIndex}.webp`;
  byId('demo-frame').alt = example.alts[frameIndex];
  byId('frame-time').textContent = currentTime;
  byId('frame-caption').textContent = example.captions[frameIndex];
  byId('demo-scrubber').value = frameIndex;
  byId('demo-scrubber').setAttribute('aria-valuetext', `${currentTime}, ${example.captions[frameIndex]}`);
  byId('playback-position').textContent = `${frameIndex + 1} / 7`;
  document.querySelectorAll('[data-frame]').forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.frame) === frameIndex)));

  const answer = byId('demo-answer');
  const state = byId('demo-state');
  const question = byId('question-block');
  question.classList.remove('pending');
  answer.classList.remove('waiting-answer');
  state.className = 'state-label';
  byId('question-time').textContent = example.questionTime;

  if (activeExample === 'memory') {
    const answered = frameIndex === 6;
    const count = Math.min(frameIndex + 1, 6);
    byId('demo-question').textContent = answered ? example.question : 'No question yet.';
    if (!answered) {
      question.classList.add('pending');
      byId('question-time').textContent = 'not yet asked';
    }
    byId('record-heading').textContent = 'Memory';

    const indices = answered ? [0,5] : Array.from({length:count}, (_, index) => index);
    byId('demo-records').innerHTML = indices.map(index => recordMarkup(example.records[index], example.times[index], answered && index === 0 ? 'record-relevant' : '')).join('');
    if (answered) {
      state.textContent = 'Response';
      answer.innerHTML = `<span class="transcript-label">OneStreamer · ${currentTime}</span><p>${escapeHTML(example.answer)}</p>`;
    } else {
      state.textContent = frameIndex === 5 ? 'Summary' : 'Observe';
      state.classList.add('observing');
      answer.classList.add('waiting-answer');
      answer.innerHTML = '<p>Waiting for a question.</p>';
    }
    byId('demo-records').scrollTop = answered ? 0 : byId('demo-records').scrollHeight;
  } else {
    byId('demo-question').textContent = example.question;
    byId('record-heading').textContent = 'Responses';
    const outputSteps = Object.keys(example.outputs).map(Number).filter(index => index <= frameIndex);

    byId('demo-records').innerHTML = outputSteps.length ? outputSteps.map(index => recordMarkup({kind:'Response',text:example.outputs[index]}, example.times[index], 'output-record')).join('') : '<p class="empty-records">No events yet.</p>';
    if (Object.hasOwn(example.outputs, frameIndex)) {
      state.textContent = 'Response';
      answer.innerHTML = `<span class="transcript-label">OneStreamer · ${currentTime}</span><p>${example.outputs[frameIndex]}</p>`;
    } else {
      state.textContent = 'Silence';
      state.classList.add('waiting');
      answer.classList.add('waiting-answer');
      answer.innerHTML = '<p>Waiting for the next event.</p>';
    }
    byId('demo-records').scrollTop = byId('demo-records').scrollHeight;
  }
  updatePlayButton();
}

function updatePlayButton() {
  const playing = replayTimer !== null;
  byId('play-symbol').textContent = playing ? 'Ⅱ' : '▶';
  byId('play-label').textContent = playing ? 'Pause' : frameIndex === 6 ? 'Replay' : 'Play';
  byId('play-demo').setAttribute('aria-label', playing ? 'Pause keyframe replay' : frameIndex === 6 ? 'Replay selected keyframes' : 'Play selected keyframes');
}

function pauseReplay() {
  clearInterval(replayTimer);
  replayTimer = null;
  updatePlayButton();
}

byId('play-demo').addEventListener('click', () => {
  if (replayTimer !== null) return pauseReplay();
  if (frameIndex === 6) frameIndex = 0;
  replayTimer = setInterval(() => {
    frameIndex = Math.min(frameIndex + 1, 6);
    if (frameIndex === 6) pauseReplay();
    renderFrame();
  }, 1800);
  renderFrame();
});
byId('demo-scrubber').addEventListener('input', event => {
  pauseReplay();
  frameIndex = Number(event.target.value);
  renderFrame();
});
byId('frame-strip').addEventListener('click', event => {
  const button = event.target.closest('[data-frame]');
  if (!button) return;
  pauseReplay();
  frameIndex = Number(button.dataset.frame);
  renderFrame();
});
document.addEventListener('visibilitychange', () => { if (document.hidden) pauseReplay(); });

function wireTabs(selector, onSelect) {
  const tabs = Array.from(document.querySelectorAll(selector));
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => {
      tabs.forEach(item => {
        item.setAttribute('aria-selected', String(item === tab));
        item.tabIndex = item === tab ? 0 : -1;
      });
      onSelect(tab);
    });
    tab.addEventListener('keydown', event => {
      let next = index;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      else if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = tabs.length - 1;
      else return;
      event.preventDefault();
      tabs[next].focus();
      tabs[next].click();
    });
  });
}

wireTabs('[data-demo]', tab => {
  pauseReplay();
  activeExample = tab.dataset.demo;
  frameIndex = activeExample === 'memory' ? 6 : 0;
  byId('demo-panel').setAttribute('aria-labelledby', tab.id);
  buildFrameStrip();
  renderFrame();
});

byId('result-charts').innerHTML = benchmarks.map(benchmark => {
  const scale = benchmark.digits === 2 ? 3.2 : 100;
  return `<article class="benchmark-chart" title="OneStreamer: ${benchmark.ours}; Qwen3-VL: ${benchmark.base}" aria-label="${benchmark.name}: OneStreamer ${benchmark.ours}, Qwen3-VL ${benchmark.base}"><h3 class="benchmark-name">${benchmark.name}</h3><p class="benchmark-metric">${benchmark.metric}</p><div class="benchmark-score"><strong>${benchmark.ours.toFixed(benchmark.digits)}</strong></div><div class="benchmark-bar-row" aria-hidden="true"><div class="benchmark-bar-track"><div class="benchmark-bar" style="width:${benchmark.ours / scale * 100}%"></div></div></div><div class="benchmark-bar-row" aria-hidden="true"><div class="benchmark-bar-track"><div class="benchmark-bar base" style="width:${benchmark.base / scale * 100}%"></div></div></div></article>`;
}).join('');

let resultGroup = 'perception';
const featured = new Set(['StreamForest','VideoChat3','AURA','MOSS-VL-Realtime','Qwen3-VL (base)','OneStreamer']);
function renderBenchmarkTable() {
  const offset = resultGroup === 'perception' ? 0 : 4;
  const group = benchmarks.slice(offset, offset + 4);
  const selectedModels = byId('all-methods').checked ? models : models.filter(model => featured.has(model.name));
  const table = byId('benchmark-table');
  table.querySelector('thead').innerHTML = `<tr><th scope="col">Method</th><th scope="col">Size</th>${group.map(benchmark => `<th scope="col">${benchmark.name} ↑<small>${benchmark.metric}</small></th>`).join('')}</tr>`;
  table.querySelector('tbody').innerHTML = selectedModels.map(model => `<tr class="${model.name === 'OneStreamer' ? 'ours-row' : model.name === 'Qwen3-VL (base)' ? 'base-row' : ''}"><th scope="row">${model.name}</th><td>${model.size}</td>${group.map((benchmark,index) => `<td>${model.scores[index + offset] === null ? '<span aria-label="Not reported">—</span>' : model.scores[index + offset].toFixed(benchmark.digits)}</td>`).join('')}</tr>`).join('');
}
wireTabs('[data-result-group]', tab => {
  resultGroup = tab.dataset.resultGroup;
  byId('benchmark-panel').setAttribute('aria-labelledby', tab.id);
  renderBenchmarkTable();
});
byId('all-methods').addEventListener('change', renderBenchmarkTable);

const figureDialog = byId('figure-dialog');
let figureTrigger = null;
document.addEventListener('click', event => {
  const trigger = event.target.closest('[data-lightbox]');
  if (!trigger) return;
  figureTrigger = trigger;
  byId('dialog-image').src = trigger.dataset.lightbox;
  byId('dialog-image').alt = trigger.dataset.caption;
  byId('dialog-caption').textContent = trigger.dataset.caption;
  byId('dialog-original').href = trigger.dataset.lightbox;
  figureDialog.showModal();
  document.body.classList.add('dialog-open');
});
byId('close-dialog').addEventListener('click', () => figureDialog.close());
figureDialog.addEventListener('click', event => {
  if (event.target !== figureDialog) return;
  const rect = figureDialog.getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) figureDialog.close();
});
figureDialog.addEventListener('close', () => {
  document.body.classList.remove('dialog-open');
  figureTrigger?.focus({preventScroll:true});
});

byId('copy-citation').addEventListener('click', async () => {
  const text = byId('bibtex').textContent;
  let copied = false;
  try {
    if (!navigator.clipboard) throw new Error('Clipboard API unavailable');
    await navigator.clipboard.writeText(text);
    copied = true;
  } catch {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.append(textarea);
    textarea.select();
    copied = document.execCommand('copy');
    textarea.remove();
  }
  const buttonLabel = byId('copy-citation').querySelector('span');
  buttonLabel.textContent = copied ? 'Copied!' : 'Select to copy';
  byId('copy-status').textContent = copied ? 'BibTeX copied to clipboard.' : 'Copy unavailable. Select and copy the citation text.';
  if (!copied) {
    const range = document.createRange();
    range.selectNodeContents(byId('bibtex'));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
  }
  setTimeout(() => { buttonLabel.textContent = 'Copy BibTeX'; }, 2500);
});

const heroVideo = byId('hero-video-player');
const heroVideoPlay = byId('hero-video-play');
heroVideo.controls = false;
heroVideoPlay.hidden = false;
heroVideoPlay.addEventListener('click', async () => {
  heroVideo.controls = true;
  heroVideoPlay.hidden = true;
  try {
    await heroVideo.play();
    heroVideo.focus({preventScroll:true});
  } catch {
    heroVideoPlay.hidden = false;
  }
});
heroVideo.addEventListener('play', () => { heroVideoPlay.hidden = true; });
heroVideo.addEventListener('ended', () => {
  heroVideoPlay.hidden = false;
  heroVideoPlay.setAttribute('aria-label', 'Replay OneStreamer video demo');
  byId('hero-video-play-label').textContent = 'Replay demo';
});

const menuToggle = document.querySelector('.menu-toggle');
function closeMenu() {
  byId('main-nav').classList.remove('is-open');
  menuToggle.setAttribute('aria-expanded','false');
  menuToggle.setAttribute('aria-label','Open navigation');
}
menuToggle.addEventListener('click', () => {
  const open = byId('main-nav').classList.toggle('is-open');
  menuToggle.setAttribute('aria-expanded', String(open));
  menuToggle.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
});
byId('main-nav').addEventListener('click', event => { if (event.target.closest('a')) closeMenu(); });
document.addEventListener('keydown', event => { if (event.key === 'Escape') closeMenu(); });
document.addEventListener('click', event => { if (!event.target.closest('.site-header')) closeMenu(); });
window.matchMedia('(min-width:581px)').addEventListener('change', event => { if (event.matches) closeMenu(); });

if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      document.querySelectorAll('#main-nav a').forEach(link => link.classList.toggle('active', link.hash === `#${entry.target.id}`));
    });
  }, {rootMargin:'-15% 0px -65% 0px'});
  ['video-demo','demo','method','dataset','results'].forEach(id => observer.observe(byId(id)));
}

buildFrameStrip();
renderFrame();
renderBenchmarkTable();

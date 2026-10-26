// How Kolam Core works: dots and gates, a kolam as a grid of digits (encoding <-> drawing, live and
// exact, spec §12), the generated data, and how the model learns to read it. Every chalkboard here is
// drawn from its encoding by the same code as the studio (fromEncoding -> compile -> drawn).

import './style.css'
import { Board, CHALK, bezierSegments } from './board'
import { compile } from './kolam/compiler'
import { drawn } from './kolam/drawn'
import { fromEncoding } from './kolam/encoding'
import { GITHUB } from './links'

const GATES = [
  { enc: ['00', '010', '00', '010', '00'], d: '0', name: 'wall', says: 'the line turns back, so the two sides stay apart', mark: [2, 0] },
  { enc: ['00', '010', '00'], d: '1', name: 'cross', says: 'the lines cross over, from one dot to the next', mark: [1, 1] },
  { enc: ['00', '020', '00'], d: '2', name: 'tube', says: 'the lines run side by side, wrapping both dots', mark: [1, 1] },
  { enc: ['0', '03', '0'], d: '3–8', name: 'horn', says: 'a petal tip grows out, pointing one way' },
] as { enc: string[]; d: string; name: string; says: string; mark?: [number, number] }[]
// a 5 x 5 kolam from the generated data (enc_demo_100k #51371): crossings inside a tube band, with petal
// horns at the sides and corners; it uses every digit, 0 to 8
const START = ['70008', '012210', '11111', '001100', '11111', '611113', '11111', '001100', '11111', '012210', '50004']
const NAMES = ['wall', 'cross', 'tube', 'horn', 'horn curving left', 'horn curving right', 'horn', 'horn curving left', 'horn curving right']
const HINT_GATE: [number, number] = [5, 3] // the gate the "tap me" ring sits on: the centre
const SYSTEMS = [
  ['random', 'grows outward from a seed, gate by gate'],
  ['substitution', 'repeats a small motif inside itself'],
  ['switching', 'flips gates in a symmetric pattern'],
  ['wfc', 'pieces tiles together like a puzzle'],
]

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <main class="rd hw">
    <header class="rd-head">
      <div class="rd-top">
        <a class="rd-pill" href="./index.html"><span aria-hidden="true">←</span> Try it</a>
        ${GITHUB}
      </div>
      <h1>How it works</h1>
      <p>A kolam looks endlessly intricate, yet it follows a few simple rules. Kolam Core turns those rules
         into numbers, makes millions of kolams from them, and learns to read them back from a photo.</p>
    </header>

    <section class="hw-sec">
      <h2><span>1</span> Dots and gates</h2>
      <p>Every kolam starts as a grid of dots. Between any two neighbouring dots there is a gap, and the
         line does one of a few things there. We call each gap a <b>gate</b>, and give each kind a digit.</p>
      <div class="hw-gates" id="gates"></div>
    </section>

    <section class="hw-sec">
      <h2><span>2</span> A kolam is a grid of digits</h2>
      <p>Write down every gate's digit, row by row, and you have the whole kolam. Go the other way, follow
         the line through each gate, and the kolam draws itself back. Both directions are exact: one
         kolam, one grid of digits.</p>
      <div class="hw-hint" id="hint"><span aria-hidden="true">👆</span> Tap on the kolam and see it change</div>
      <div class="hw-play">
        <figure class="hw-board"><div class="rd-frame"><canvas id="play" class="board"></canvas>
          <span class="hw-tap" id="tap" aria-hidden="true"></span></div>
          <figcaption id="playcap"></figcaption></figure>
        <div class="hw-digits">
          <div id="grid" class="hw-grid" aria-label="the kolam as digits: tap one to change it"></div>
          <div class="hw-legend"><span class="d0">0 wall</span><span class="d1">1 cross</span><span class="d2">2 tube</span>
            <span class="d3">3–5 horn → or ↓</span><span class="d6">6–8 horn ← or ↑</span>
            <small>(each horn: plain, curving left, curving right)</small></div>
          <button class="rd-btn" id="reset">↺ start over</button>
        </div>
      </div>
      <div class="hw-math">
        <p>For a kolam with <b>R</b> rows and <b>C</b> columns of dots, the digits come in
           <b>2R + 1</b> rows, alternating <b>C</b> gates (between dots one above the other) and
           <b>C + 1</b> gates (between dots side by side):</p>
        <p class="hw-formula">gates = R(C + 1) + (R + 1)C</p>
        <p>A 3 × 3 kolam has 24 gates, a 5 × 5 has 60, a 20 × 20 has 840. Gates on the outer edge can only
           be a wall or a horn pointing outward, which is why so many digits there are 0.</p>
      </div>
    </section>

    <section class="hw-sec">
      <h2><span>3</span> Millions of kolams to learn from</h2>
      <p>To learn, a model needs kolams together with their exact digits. We make them: four small programs
         grow kolams from the rules above, and every kolam is checked against the kolam rules before it is
         kept. Around two million so far, from 3 × 3 up to 20 × 20 dots.</p>
      <div class="hw-four" id="systems"></div>
      <p>Then each kolam is drawn as if photographed: chalk or rice-flour lines, cement, red oxide or tiled
         floors, a tilted camera, sunlight and glare. The digits never change, only the look.</p>
      <div class="hw-looks" id="looks"></div>
    </section>

    <section class="hw-sec">
      <h2><span>4</span> The model that reads</h2>
      <p>An AI chatbot writes its reply one word at a time. Kolam Core works the same way, one digit at a
         time. It has two parts: the <b>eyes</b> (a ResNet) turn the photo into 576 small pieces, and the
         <b>writer</b> (a transformer decoder, the same kind of block that is stacked inside an LLM) writes
         the digits, looking at the photo and at what it has already written. The writer is drawn here the way
         the transformer paper draws it, bottom to top, with this model’s real sizes.</p>
      <div class="tf" id="tf">
        <svg class="tf-link" aria-hidden="true"><path id="tfLink"/></svg>
        <div class="tf-col">
          <div class="tf-tag">the eyes · once per photo</div>
          <div class="tf-step" data-s="0"><div class="tf-photo"><img class="tf-src" alt="a kolam photo"><i></i></div>
            <small><b>photo</b>384 × 384 pixels</small></div>
          <small class="tf-head"><b>ResNet-34, up to its third stage</b>27 layers of filters, each seeing a bigger patch of the photo;
            the grid shrinks while each spot holds more numbers</small>
          <svg class="tf-svg tf-enc" viewBox="0 0 380 438" role="img" aria-label="ResNet-34, up to its third stage, with each part and its feature-map size">
            <defs><marker id="eh" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#9CA3AF"/></marker>
              <marker id="ehr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#5d7a72"/></marker></defs>
            <path class="tf-wire" d="M135,2 V19" marker-end="url(#eh)"/>
            <path class="tf-wire" d="M135,42 V55" marker-end="url(#eh)"/>
            <path class="tf-wire" d="M135,90 V103" marker-end="url(#eh)"/>
            <path class="tf-wire" d="M135,126 V139" marker-end="url(#eh)"/>
            <path class="tf-wire" d="M135,174 V187" marker-end="url(#eh)"/>
            <path class="tf-wire" d="M135,228 V241" marker-end="url(#eh)"/>
            <path class="tf-wire" d="M135,282 V295" marker-end="url(#eh)"/>
            <path class="tf-wire" d="M135,336 V349" marker-end="url(#eh)"/>
            <path class="tf-wire" d="M135,380 V393" marker-end="url(#eh)"/>
            <path class="tf-wire" d="M135,418 V437" marker-end="url(#eh)"/>
            <g data-e="0"><rect class="tf-norm" x="50" y="20" width="170" height="22" rx="5"/><text class="tf-sm" x="135" y="35.5">Normalize (ImageNet mean, std)</text></g>
            <g data-e="1"><rect class="tf-conv" x="30" y="56" width="210" height="34" rx="8"/><text x="135" y="71.0">Conv 7 × 7 · 64 filters</text><text class="tf-sm" x="135" y="85.0">stride 2</text><rect class="tf-map" x="262" y="53.0" width="40.0" height="40.0" rx="2"/><text class="tf-side" x="306" y="72.0">192 × 192</text><text class="tf-side" x="306" y="85.0">× 64</text></g>
            <g data-e="2"><rect class="tf-norm" x="50" y="104" width="170" height="22" rx="5"/><text class="tf-sm" x="135" y="119.5">BatchNorm + ReLU</text></g>
            <g data-e="3"><rect class="tf-lin" x="30" y="140" width="210" height="34" rx="8"/><text x="135" y="155.0">Max-Pool 3 × 3</text><text class="tf-sm" x="135" y="169.0">stride 2</text><rect class="tf-map" x="262" y="145.5" width="23.0" height="23.0" rx="2"/><text class="tf-side" x="306" y="156.0">96 × 96</text><text class="tf-side" x="306" y="169.0">× 64</text></g>
            <g data-e="4"><rect class="tf-conv tf-back" x="36" y="182" width="210" height="40" rx="8"/><rect class="tf-conv tf-back" x="33" y="185" width="210" height="40" rx="8"/><rect class="tf-conv" x="30" y="188" width="210" height="40" rx="8"/><text x="135" y="206.0">layer1 · 3 residual blocks</text><text class="tf-sm" x="135" y="220.0">64 channels</text><rect class="tf-map" x="262" y="196.5" width="23.0" height="23.0" rx="2"/><text class="tf-side" x="306" y="207.0">96 × 96</text><text class="tf-side" x="306" y="220.0">× 64</text></g>
            <g data-e="5"><rect class="tf-conv tf-back" x="36" y="236" width="210" height="40" rx="8"/><rect class="tf-conv tf-back" x="33" y="239" width="210" height="40" rx="8"/><rect class="tf-conv" x="30" y="242" width="210" height="40" rx="8"/><text x="135" y="260.0">layer2 · 4 residual blocks</text><text class="tf-sm" x="135" y="274.0">128 channels, stride 2</text><rect class="tf-map" x="262" y="254.75" width="14.5" height="14.5" rx="2"/><text class="tf-side" x="306" y="261.0">48 × 48</text><text class="tf-side" x="306" y="274.0">× 128</text></g>
            <g data-e="6"><rect class="tf-conv tf-back" x="36" y="290" width="210" height="40" rx="8"/><rect class="tf-conv tf-back" x="33" y="293" width="210" height="40" rx="8"/><rect class="tf-conv" x="30" y="296" width="210" height="40" rx="8"/><text x="135" y="314.0">layer3 · 6 residual blocks</text><text class="tf-sm" x="135" y="328.0">256 channels, stride 2</text><rect class="tf-map" x="262" y="310.875" width="10.25" height="10.25" rx="2"/><text class="tf-side" x="306" y="315.0">24 × 24</text><text class="tf-side" x="306" y="328.0">× 256</text></g>
            <g data-e="7"><rect class="tf-conv" x="30" y="350" width="210" height="30" rx="8"/><text x="135" y="369.5">Conv 1 × 1 · 256</text></g>
            <g data-e="8"><circle class="tf-emb" cx="135" cy="406.0" r="12.0"/><text class="tf-o" x="135" y="411.0">+</text><text class="tf-side" x="153" y="404.0">2D position code:</text><text class="tf-side" x="153" y="417.0">where each patch sits</text></g>
            <g data-e="7"><path class="tf-res" d="M240,316.0 H254 V346" marker-end="url(#ehr)"/><rect class="tf-lin" x="254" y="348" width="122" height="38" rx="8"/><text class="tf-sm" x="315" y="364">mean-pool → Linear</text><text class="tf-sm" x="315" y="378">“is it a kolam?”</text></g>
          </svg>
          <div class="tf-step tf-img" data-s="0 3"><div class="tf-tokens" id="tfImg"></div>
            <small><b>576 image tokens</b>a 24 × 24 grid: each patch of the photo is now a list of 256 numbers</small></div>
          <small class="tf-head tf-inset">inside every residual block</small>
          <svg class="tf-svg tf-block1" viewBox="0 0 380 64" role="img" aria-label="inside one residual block"><path class="tf-wire" d="M2,44 H14"/><path class="tf-wire" d="M14,44 H25" marker-end="url(#eh)"/><rect class="tf-conv" x="26" y="32" width="58" height="24" rx="6"/><text class="tf-sm" x="55.0" y="48">Conv 3×3</text><path class="tf-wire" d="M86,44 H97" marker-end="url(#eh)"/><rect class="tf-norm" x="98" y="32" width="58" height="24" rx="6"/><text class="tf-sm" x="127.0" y="48">BN · ReLU</text><path class="tf-wire" d="M158,44 H169" marker-end="url(#eh)"/><rect class="tf-conv" x="170" y="32" width="58" height="24" rx="6"/><text class="tf-sm" x="199.0" y="48">Conv 3×3</text><path class="tf-wire" d="M230,44 H241" marker-end="url(#eh)"/><rect class="tf-norm" x="242" y="32" width="58" height="24" rx="6"/><text class="tf-sm" x="271.0" y="48">BN</text><path class="tf-wire" d="M302,44 H311" marker-end="url(#eh)"/><circle class="tf-add" cx="322" cy="44" r="10"/><text class="tf-o" x="322" y="49">+</text><path class="tf-res" d="M14,44 V16 H322 V33" marker-end="url(#ehr)"/><text class="tf-side" x="114.0" y="11">skip: the input, added back</text><path class="tf-wire" d="M332,44 H344" marker-end="url(#eh)"/><text class="tf-sm" x="362" y="48">ReLU</text></svg>
        </div>
        <div class="tf-col">
          <div class="tf-tag">the writer · once per digit</div>
          <div class="tf-step tf-out" data-s="6 7"><div class="rd-frame"><canvas id="tfBoard" class="board"></canvas></div>
            <small><b>the kolam so far</b>each digit is a gate; a row of dots is drawn once its digits are done</small></div>
          <div class="tf-step" data-s="6"><div class="tf-pick" id="tfPick"></div></div>
          <div class="tf-step" data-s="5 6"><div class="hw-bars" id="tfBars"></div>
            <small><b>11 scores</b>the raw score (logit) for each token: 0–8, “|” ends a row, ■ ends the kolam.
            Softmax turns them into chances; the kolam rules strike out tokens that can’t make a kolam</small></div>
          <svg class="tf-svg" viewBox="0 72 440 588" role="img" aria-label="the transformer decoder, drawn as in the paper, with this model's sizes">
            <defs><marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#9CA3AF"/></marker>
              <marker id="ahr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#5d7a72"/></marker>
              <marker id="ahk" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#FACC15"/></marker></defs>
            <rect class="tf-n" x="96" y="182" width="290" height="376" rx="14"/><text class="tf-nx" x="410" y="376.0">× 4</text>
            <path class="tf-res" d="M230,558.0 H356 V450.0 H242.0" marker-end="url(#ahr)"/>
            <path class="tf-res" d="M230,432.0 H356 V324.0 H242.0" marker-end="url(#ahr)"/>
            <path class="tf-res" d="M230,306.0 H356 V200.0 H242.0" marker-end="url(#ahr)"/>
            <path class="tf-wire" d="M230,604 V589" marker-end="url(#ah)"/>
            <path class="tf-wire" d="M230,566 V551" marker-end="url(#ah)"/>
            <path class="tf-wire" d="M230,530 V515" marker-end="url(#ah)"/>
            <path class="tf-wire" d="M230,476 V461" marker-end="url(#ah)"/>
            <path class="tf-wire" d="M230,440 V425" marker-end="url(#ah)"/>
            <path class="tf-wire" d="M230,404 V389" marker-end="url(#ah)"/>
            <path class="tf-wire" d="M230,350 V335" marker-end="url(#ah)"/>
            <path class="tf-wire" d="M230,314 V299" marker-end="url(#ah)"/>
            <path class="tf-wire" d="M230,278 V263" marker-end="url(#ah)"/>
            <path class="tf-wire" d="M230,226 V211" marker-end="url(#ah)"/>
            <path class="tf-wire" d="M230,190 V175" marker-end="url(#ah)"/>
            <path class="tf-wire" d="M230,154 V139" marker-end="url(#ah)"/>
            <path class="tf-wire" d="M230,104 V78" marker-end="url(#ah)"/>
            <g data-s="1"><rect class="tf-emb" x="120" y="604.0" width="220" height="36" rx="8"/><text x="230" y="619.0">digit + row + column embeddings</text><text class="tf-sm" x="230" y="633.0">added up: 256 numbers per token</text><circle class="tf-add" cx="230" cy="577.0" r="11.0"/><text class="tf-o" x="230" y="582.0">+</text></g>
            <g data-s="2"><rect class="tf-norm" x="150" y="530" width="160" height="20" rx="5"/><text class="tf-sm" x="230" y="545.0">Norm</text><rect class="tf-att" x="120" y="476" width="220" height="38" rx="8"/><text x="230" y="492.0">Masked Multi-Head</text><text class="tf-sm" x="230" y="507.0">Self-Attention · 8 heads</text><circle class="tf-add" cx="230" cy="450.0" r="10.0"/><text class="tf-o" x="230" y="455.0">+</text><text class="tf-side" x="104" y="472" text-anchor="start">Q K V: the digits so far</text></g>
            <g data-s="3"><rect class="tf-norm" x="150" y="404" width="160" height="20" rx="5"/><text class="tf-sm" x="230" y="419.0">Norm</text><rect class="tf-att" x="120" y="350" width="220" height="38" rx="8"/><text x="230" y="366.0">Multi-Head Cross-Attention</text><text class="tf-sm" x="230" y="381.0">8 heads</text><circle class="tf-add" cx="230" cy="324.0" r="10.0"/><text class="tf-o" x="230" y="329.0">+</text><path class="tf-kv" d="M4,369.0 H118" marker-end="url(#ahk)"/><text class="tf-side tf-kvt" x="6" y="361.0" text-anchor="start">K V: 576 image</text><text class="tf-side tf-kvt" x="6" y="385.0" text-anchor="start">tokens (the eyes)</text></g>
            <g data-s="4"><rect class="tf-norm" x="150" y="278" width="160" height="20" rx="5"/><text class="tf-sm" x="230" y="293.0">Norm</text><rect class="tf-ff" x="120" y="226" width="220" height="36" rx="8"/><text x="230" y="241.0">Feed Forward</text><text class="tf-sm" x="230" y="256.0">256 → 1024 → 256</text><circle class="tf-add" cx="230" cy="200.0" r="10.0"/><text class="tf-o" x="230" y="205.0">+</text></g>
            <g data-s="5"><rect class="tf-norm" x="150" y="154" width="160" height="20" rx="5"/><text class="tf-sm" x="230" y="169.0">Norm</text><rect class="tf-lin" x="120" y="104" width="220" height="34" rx="8"/><text x="230" y="118.0">Linear</text><text class="tf-sm" x="230" y="133.0">256 → 11 scores</text></g>
          </svg>
          <p class="tf-note">dashed lines: residual connections, each part adds its result to what came in</p>
          <div class="tf-step" data-s="1"><div class="tf-chips" id="tfChips"></div>
            <small><b>the digits so far, as tokens</b>it starts from ▸ and reads its own answer so far</small></div>
          <div class="tf-loop" data-s="7">↻ the token it wrote joins the inputs at the bottom; all 4 blocks run again for the next one</div>
        </div>
      </div>
      <p>That loop is the whole of the reading: 72 passes for this 5 × 5 kolam, about 300 for a 12 × 12 one.
         Each pass runs the 4 blocks over every digit written so far, so the browser keeps the work small by
         sending only those digits, never empty space.</p>
    </section>

    <section class="hw-sec">
      <h2><span>5</span> Watch it write</h2>
      <p>This is the real model reading the photo, step by step: each digit it wrote, how sure it was, and
         where on the photo it looked while writing it. Step through it, or press play.</p>
      <div class="wr">
        <figure class="wr-pane"><div class="wr-photo"><img class="tf-src" alt="the photo being read"><canvas id="wrLook"></canvas></div>
          <figcaption>where it looks<small>bright: the patches it attends to most</small></figcaption></figure>
        <figure class="wr-pane"><div class="rd-frame"><canvas id="wrBoard" class="board"></canvas></div>
          <figcaption>the kolam so far<small>drawn as each row of dots is finished</small></figcaption></figure>
        <figure class="wr-pane"><div class="hw-grid wr-grid" id="wrGrid"></div>
          <figcaption>the digits<small>written left to right, row by row</small></figcaption></figure>
      </div>
      <div class="hw-bars wr-bars" id="wrBars"></div>
      <p class="wr-say" id="wrSay"></p>
      <div class="wr-ctl">
        <button class="rd-btn" id="wrReset" aria-label="back to the start">⏮</button>
        <button class="rd-btn" id="wrBack" aria-label="one step back">◀</button>
        <button class="rd-btn primary" id="wrPlay">▶ Play</button>
        <button class="rd-btn" id="wrNext" aria-label="one step forward">▶|</button>
        <input type="range" id="wrStep" min="0" max="1" value="0" aria-label="step">
      </div>
    </section>

    <section class="hw-sec">
      <h2><span>6</span> How it learned</h2>
      <div class="hw-train">
        <div class="hw-loop" aria-label="the training loop">
          <div><b>1</b> a generated photo, with its true digits</div>
          <div><b>2</b> the model writes its guess</div>
          <div><b>3</b> each digit is compared with the truth</div>
          <div><b>4</b> the model is nudged to do a little better</div>
          <span class="hw-again">↻ and again, with the next photo</span>
        </div>
        <figure class="hw-chart"><svg id="chart" viewBox="0 0 360 220" role="img"
          aria-label="training: mistakes fall while exact readings rise"></svg>
          <figcaption id="chartcap"></figcaption></figure>
      </div>
      <p>This is the real training record of the model on the first page: one night on a gaming laptop's
         GPU, 2.4 million photos. The yellow line is how wrong its guesses were; the green dots are how many
         kolams it had <b>never seen</b> it read exactly, checked along the way.</p>
    </section>

    <section class="hw-sec">
      <h2><span>7</span> All inside Chrome</h2>
      <div class="hw-chrome">
        <div><span class="hw-icon">⬇</span><b>once</b><small>the model downloads to your browser the first time</small></div>
        <i class="hw-arrow">→</i>
        <div><span class="hw-icon">⚙</span><b>in Chrome</b><small>it runs on WebAssembly, in a background thread</small></div>
        <i class="hw-arrow">→</i>
        <div><span class="hw-icon">🔒</span><b>on your device</b><small>your photo never leaves it</small></div>
      </div>
      <p>The models are optimized to run in Chrome: converted to a compact format (ONNX) and run by
         WebAssembly in a background thread, so everything happens inside Chrome itself. There is no
         server doing the work, and the page stays smooth while the model reads.</p>
    </section>

    <footer class="rd-foot">
      <p class="rd-links"><a class="rd-pill" href="./index.html"><span aria-hidden="true">←</span> Try it on a kolam</a> ${GITHUB}</p>
    </footer>
  </main>`

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T

/** Draw an encoding on a board; `mark` rings one gate (encoding row, digit index). */
function draw(board: Board, enc: string[], mark?: [number, number]): void {
  const h = fromEncoding(enc)
  const k = compile(h)
  board.layout(h.rows, h.cols, 1)
  board.clear()
  board.dots(CHALK.main, Math.max(2, board.s * 0.14), new Set(Array.from({ length: h.rows * h.cols }, (_, n) => `${Math.floor(n / h.cols)},${n % h.cols}`)))
  const w = Math.max(1.5, board.s * 0.14)
  drawn(k).strokes.forEach((P, si) => bezierSegments(board, P).forEach((seg, i) => board.chalk(seg, CHALK.main, w, si * 1009 + i * 7 + 1)))
  if (mark) {
    const [x, y] = board.px(gapAt(mark[0], mark[1]))
    const ctx = board.ctx
    ctx.strokeStyle = CHALK.highlight
    ctx.lineWidth = 2.5
    ctx.beginPath()
    ctx.arc(x, y, board.s * 0.55, 0, Math.PI * 2)
    ctx.stroke()
  }
}

/** The lattice point of digit k in encoding row i (odd rows: left/right gaps at even columns). */
const gapAt = (i: number, k: number): [number, number] => [i, i % 2 ? 2 * k : 2 * k + 1]

// 1 · the gates, one small board each
$('gates').innerHTML = GATES.map((g, n) => `<figure class="hw-gate"><div class="rd-frame"><canvas id="g${n}" class="board"></canvas></div>
  <figcaption><b>${g.d}</b> ${g.name}<small>${g.says}</small></figcaption></figure>`).join('')
const gateBoards = GATES.map((_, n) => new Board($<HTMLCanvasElement>(`g${n}`), ''))
const drawGates = () => GATES.forEach((g, n) => draw(gateBoards[n], g.enc, g.mark))

// 2 · the playground: a 5 x 5 kolam and its digits, linked both ways
let enc = [...START]
let mark: [number, number] | undefined
let touched = false // until the first tap, a ring and a hint say "tap me"
const play = new Board($<HTMLCanvasElement>('play'), '')
const last = () => enc.length - 1

/** The digits gate (i, k) can hold: inside, any of 0-8; on the outer edge, a wall or a horn pointing
 *  out of the grid (top and left edges: 6-8, pointing up / left; bottom and right: 3-5). */
function choices(i: number, k: number): number[] {
  const out = (up: boolean) => [0, ...(up ? [6, 7, 8] : [3, 4, 5])]
  if (i % 2 === 0) return i === 0 ? out(true) : i === last() ? out(false) : [0, 1, 2, 3, 4, 5, 6, 7, 8]
  return k === 0 ? out(true) : k === enc[i].length - 1 ? out(false) : [0, 1, 2, 3, 4, 5, 6, 7, 8]
}

/** What a digit means at gate (i, k), in words. */
function named(d: number, i: number): string {
  if (d < 3) return NAMES[d]
  const way = i % 2 ? (d < 6 ? 'right' : 'left') : d < 6 ? 'down' : 'up'
  return `${NAMES[d]}, pointing ${way}`
}

function render(): void {
  draw(play, enc, mark)
  $('grid').innerHTML = enc
    .map((row, i) => `<div class="hw-row ${i % 2 ? 'lr' : 'ud'}">${[...row].map((d, k) => {
      const hint = !touched && i === HINT_GATE[0] && k === HINT_GATE[1] ? ' pulse' : ''
      const on = mark && mark[0] === i && mark[1] === k ? ' on' : ''
      const edge = choices(i, k).length < 9 ? ' edge' : ''
      return `<button data-i="${i}" data-k="${k}" class="d${d}${on}${edge}${hint}" aria-label="gate ${i + 1}.${k + 1}: ${named(+d, i)}">${d}</button>`
    }).join('')}</div>`)
    .join('')
  $('grid').querySelectorAll<HTMLButtonElement>('button').forEach((b) => (b.onclick = () => flip(+b.dataset.i!, +b.dataset.k!)))
  $<HTMLButtonElement>('reset').disabled = enc.join() === START.join() // nothing to undo yet
  const h = fromEncoding(enc)
  const lines = compile(h).strokes.length
  $('playcap').textContent = mark
    ? `that gate is now a ${named(+enc[mark[0]][mark[1]], mark[0])}: ${lines} line${lines === 1 ? '' : 's'} in all`
    : `${h.rows} × ${h.cols} dots, ${lines} line${lines === 1 ? '' : 's'}`
  // the "tap me" ring sits on the hint gate until the first tap
  const tap = $('tap')
  tap.hidden = touched
  const [x, y] = play.px(gapAt(HINT_GATE[0], HINT_GATE[1]))
  Object.assign(tap.style, { left: `${x}px`, top: `${y}px`, width: `${play.s * 1.3}px`, height: `${play.s * 1.3}px` })
}

/** Step gate (i, k) to its next digit that can be drawn (one that can't is skipped). */
function flip(i: number, k: number): void {
  if (!touched) ((touched = true), ($('hint').innerHTML = 'Keep tapping, and watch the kolam and its numbers change together'))
  const before = enc[i]
  const opts = choices(i, k)
  const at = opts.indexOf(+before[k])
  for (let n = 1; n <= opts.length; n++) {
    const d = opts[(at + n) % opts.length]
    enc[i] = before.slice(0, k) + d + before.slice(k + 1)
    try {
      compile(fromEncoding(enc)) // drawable?
      break
    } catch {
      enc[i] = before
    }
  }
  mark = [i, k]
  render()
}

$<HTMLCanvasElement>('play').addEventListener('click', (e) => {
  const r = (e.target as HTMLCanvasElement).getBoundingClientRect()
  const [x, y] = [e.clientX - r.left, e.clientY - r.top]
  let best: [number, number] | null = null
  let dist = Infinity
  enc.forEach((row, i) => [...row].forEach((_, k) => {
    const [gx, gy] = play.px(gapAt(i, k))
    const d = Math.hypot(gx - x, gy - y)
    if (d < dist) (dist = d), (best = [i, k])
  }))
  if (best && dist < play.s * 1.1) flip(best[0], best[1])
})
/** Back to the original kolam: the board comes into view and the kolam fades back in. */
$('reset').onclick = () => {
  enc = [...START]
  mark = undefined
  const c = $('play')
  c.scrollIntoView({ behavior: 'smooth', block: 'center' })
  c.classList.remove('hw-fade')
  void c.offsetWidth // restart the fade
  c.classList.add('hw-fade')
  render()
}

// 3 · the data
$('systems').innerHTML = SYSTEMS.map(([s, says]) => `<figure><img src="./how/system_${s}.jpg" alt="a kolam made by the ${s} system" loading="lazy">
  <figcaption><b>${s === 'wfc' ? 'tiles' : s}</b><small>${says}</small></figcaption></figure>`).join('')
$('looks').innerHTML = [1, 2, 3, 4, 5, 6].map((n) => `<img src="./how/look_${n}.jpg" alt="the same kolam, look ${n}" loading="lazy">`).join('')

// 4 + 5 · the model reading example 1 for real (public/how/decode.json, scripts/how_assets.py): every token it
// wrote, its 11 scores, the ones the kolam rules allowed, and where it looked (cross-attention, 0-255 per patch)
type Step = { tok: number; p: number[]; l: number[]; ok: number[]; look: string }
type Decode = { photo: string; grid: [number, number]; encoding: string[]; exact: boolean; steps: Step[] }
const TOKS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '|', '■']
const BAR = 9
const calm = matchMedia('(prefers-reduced-motion: reduce)').matches
let dec: Decode | null = null
let looks: Uint8Array[] = []

/** The digit rows written in the first t steps (a row counts once its "|" is written), and the row being written. */
function written(t: number): { rows: string[]; open: string } {
  const rows: string[] = []
  let open = ''
  for (const st of dec!.steps.slice(0, t)) {
    if (st.tok === BAR) (rows.push(open), (open = ''))
    else if (st.tok < BAR) open += TOKS[st.tok]
  }
  return { rows, open }
}

/** The exact chances of step `st` (softmax of its raw scores). */
function chances(st: Step): number[] {
  const m = Math.max(...st.l)
  const e = st.l.map((v) => Math.exp(v - m))
  const z = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / z)
}
/** A chance as a person would write it: 95.39%, 99.987%, 0.04%; never a rounded-up 100%. */
function pct(p: number): string {
  if (p < 0.0001) return '<0.01%'
  const d = p > 0.9999 ? 4 : p > 0.999 ? 3 : 2
  return `${Math.min(100 * p, 100 - 10 ** -d).toFixed(d)}%`
}
/** "95.39% sure · next best: | at 4.60%" */
function sureOf(st: Step): string {
  const p = chances(st)
  const next = p.map((v, j) => [v, j]).filter(([, j]) => j !== st.tok).sort((a, b) => b[0] - a[0])[0]
  return `${pct(p[st.tok])} sure · next best: <b>${TOKS[next[1]]}</b> at ${pct(next[0])}`
}

/** Step `st` as 11 bars: its chances (softmax), or with `raw` its raw scores (logits, lowest to highest);
 *  with `rules`, the tokens the kolam rules forbid are struck out and the one written is lit. */
function bars(el: HTMLElement, st: Step | undefined, rules: boolean, raw = false): void {
  const [lo, hi] = st ? [Math.min(...st.l), Math.max(...st.l)] : [0, 1]
  el.innerHTML = TOKS.map((t, j) => {
    const p = st ? chances(st)[j] : 0
    const h = !st ? 0 : raw ? 0.08 + 0.92 * (st.l[j] - lo) / (hi - lo || 1) : p
    const no = rules && st && !st.ok.includes(j) ? ' no' : ''
    const win = rules && st && st.tok === j ? ' win' : ''
    const say = raw ? st?.l[j].toFixed(1) : pct(p)
    return `<div class="hw-bar${no}${win}"><span style="height:${Math.max(2, h * 100)}%"></span>` +
      (raw ? `<b>${t}</b><i>${say ?? ''}</i>` : `<em>${win ? say : ''}</em><b>${t}</b>`) + '</div>'
  }).join('')
}

/** Where step `st` looked, on a canvas over the photo: the patches it attended to glow, the rest dims. */
function lookAt(cv: HTMLCanvasElement, t: number): void {
  const [gh, gw] = dec!.grid
  const r = cv.getBoundingClientRect()
  const dpr = devicePixelRatio || 1
  ;[cv.width, cv.height] = [Math.round(r.width * dpr), Math.round(r.height * dpr)]
  const c = cv.getContext('2d')!
  c.clearRect(0, 0, cv.width, cv.height)
  if (!t) return
  const a = looks[t - 1]
  const [w, h] = [cv.width / gw, cv.height / gh]
  for (let n = 0; n < gh * gw; n++) {
    const v = a[n] / 255
    const [x, y] = [(n % gw) * w, Math.floor(n / gw) * h]
    c.fillStyle = `rgba(11,31,27,${0.62 * (1 - Math.sqrt(v))})`
    c.fillRect(x, y, w + 0.5, h + 0.5)
    if (v > 0.08) ((c.fillStyle = `rgba(250,204,21,${0.85 * v})`), c.fillRect(x, y, w + 0.5, h + 0.5))
  }
}

/** The kolam after t steps on `board`: every dot, the lines of the rows of dots finished so far, and the
 *  gate of the digit just written ringed. Returns that gate (digit row, index), if a digit. */
function soFar(board: Board, t: number): [number, number] | null {
  const { rows, open } = written(t)
  const st = t ? dec!.steps[t - 1] : undefined
  const h = fromEncoding(dec!.encoding)
  board.layout(h.rows, h.cols, 1)
  board.clear()
  board.dots(CHALK.main, Math.max(2, board.s * 0.14), new Set(Array.from({ length: h.rows * h.cols }, (_, n) => `${Math.floor(n / h.cols)},${n % h.cols}`)))
  const done = rows.length % 2 ? rows.length : rows.length - 1
  if (done >= 3) {
    const k = compile(fromEncoding(rows.slice(0, done)))
    const w = Math.max(1.5, board.s * 0.14)
    drawn(k).strokes.forEach((P, si) => bezierSegments(board, P).forEach((seg, i) => board.chalk(seg, CHALK.main, w, si * 1009 + i * 7 + 1)))
  }
  const at: [number, number] | null = st && st.tok < BAR ? [rows.length, open.length - 1] : null
  if (at) {
    const [x, y] = board.px(gapAt(at[0], at[1]))
    const ctx = board.ctx
    ctx.strokeStyle = CHALK.highlight
    ctx.lineWidth = 2.5
    ctx.beginPath()
    ctx.arc(x, y, board.s * 0.55, 0, Math.PI * 2)
    ctx.stroke()
  }
  return at
}

// 4 · the transformer: a light walks through the stages, one pass per digit, with the real tokens and scores
const tfSteps = [...document.querySelectorAll<HTMLElement>('#tf [data-s], #tf [data-e]')]
const PARTS = 9 // the eyes' parts (data-e 0-8), lit one after another when a photo comes in
const SLOW = 3 // the first 3 digits go through every stage slowly; the rest are written quickly
const tfBoard = new Board($<HTMLCanvasElement>('tfBoard'), '')
let stage = 0, pass = 0, sub = -1

/** Is `el` lit now? Stage 0 (a new photo) walks the eyes: the photo, each part in turn, then the tokens. */
function lit(el: HTMLElement): boolean {
  if (calm) return true
  if (el.dataset.e !== undefined) return stage === 0 && +el.dataset.e === sub
  const ss = el.dataset.s!.split(' ')
  if (stage === 0 && ss.includes('0')) return el.classList.contains('tf-img') ? sub >= PARTS : sub < 0
  return ss.includes(String(stage))
}

/** The wire from the eyes to the writer: the 576 image tokens into cross-attention's K, V. Side by side it
 *  curves across; stacked (phones) it runs down the left edge. It flows while cross-attention reads it. */
function tfLink(): void {
  const box = $('tf').getBoundingClientRect()
  const a = $('tfImg').getBoundingClientRect()
  const b = document.querySelector('#tf .tf-kv')!.getBoundingClientRect()
  const [ex, ey] = [b.left - box.left, b.top + b.height / 2 - box.top]
  let d: string
  if (a.right < b.left) {
    const [sx, sy] = [a.right - box.left + 4, a.top + a.height / 2 - box.top]
    d = `M${sx},${sy} C${sx + 70},${sy} ${ex - 70},${ey} ${ex},${ey}`
  } else {
    const card = $('tfImg').closest('.tf-col')!.getBoundingClientRect() // below the whole eyes card
    const [sx, sy] = [a.left + a.width / 2 - box.left, card.bottom - box.top]
    d = `M${sx},${a.bottom - box.top + 2} M${sx},${sy} V${sy + 9} H5 V${ey} H${ex}`
  }
  $('tfLink').setAttribute('d', d)
  $('tfLink').classList.toggle('hot', stage === 3)
}

function tfShow(): void {
  if (!dec) return
  tfLink()
  tfSteps.forEach((el) => el.classList.toggle('lit', lit(el)))
  const st = dec.steps[pass]
  // the tokens it reads: BOS, then the digits so far (the last 9 of them)
  const so = ['▸', ...dec.steps.slice(0, pass).map((x) => TOKS[x.tok])]
  const shown = so.slice(-10)
  $('tfChips').innerHTML = (so.length > shown.length ? '<i>…</i>' : '') +
    shown.map((t, k) => `<i class="d${t}${k === shown.length - 1 && stage >= 1 ? ' new' : ''}">${t}</i>`).join('')
  $('tfImg').classList.toggle('looking', stage === 3)
  $('tfImg').querySelectorAll<HTMLElement>('i').forEach((cell, n) => {
    cell.style.opacity = stage === 3 ? String(0.12 + 0.88 * Math.sqrt(looks[pass][n] / 255)) : ''
    cell.classList.toggle('hot', stage === 3 && looks[pass][n] > 100)
  })
  bars($('tfBars'), stage >= 5 ? st : undefined, stage >= 6, true)
  soFar(tfBoard, stage === 0 ? 0 : pass + (stage >= 6 ? 1 : 0))
  $('tfPick').innerHTML = stage >= 6
    ? `writes <b>${TOKS[st.tok]}</b>, the best score the rules allow: ${sureOf(st)}`
    : '&nbsp;'
}
function tfTick(): void {
  if (!dec) return
  if (stage === 0 && sub < PARTS) {
    sub++
    tfShow()
    return void setTimeout(tfTick, 600)
  }
  stage++
  if (stage > 7) {
    pass = (pass + 1) % dec.steps.length
    if (pass === 0) ((stage = 0), (sub = -1)) // a new photo: the eyes look once
    else stage = 1
  }
  tfShow()
  const fast = pass >= SLOW
  setTimeout(tfTick, fast ? 140 : stage === 3 ? 1900 : 850) // a longer look at where it looks
}

// 5 · watch it write: photo + where it looks, the kolam so far, the digits; step by step
let t = 0
let timer = 0
const wrBoard = new Board($<HTMLCanvasElement>('wrBoard'), '')
function wrShow(): void {
  if (!dec) return
  const N = dec.steps.length
  const final = dec.encoding
  const { rows, open } = written(t)
  const st = t ? dec.steps[t - 1] : undefined
  const at = soFar(wrBoard, t)
  // the digits: the final shape, filled in as far as written
  $('wrGrid').innerHTML = final.map((row, i) => `<div class="hw-row ${i % 2 ? 'lr' : 'ud'}">${[...row].map((_, k) => {
    const d = i < rows.length ? rows[i][k] : i === rows.length ? open[k] : undefined
    const on = at && at[0] === i && at[1] === k ? ' on' : ''
    return d === undefined ? '<span class="todo">·</span>' : `<span class="d${d}${on}">${d}</span>`
  }).join('')}</div>`).join('')
  lookAt($<HTMLCanvasElement>('wrLook'), t)
  bars($('wrBars'), st, true, true)
  $('wrSay').innerHTML = !st
    ? `Press <b>play</b>, or step through the ${N} steps one by one.`
    : `Step ${t} of ${N}: ` + (st.tok === 10
      ? `<b>■</b>, the kolam is complete.${dec.exact ? ' Every digit matches the real kolam.' : ''}`
      : st.tok === BAR
        ? `<b>|</b>, the end of digit row ${rows.length}.`
        : `wrote <b>${TOKS[st.tok]}</b>, a ${named(st.tok, rows.length)}.`) + `<br><small>${sureOf(st)}</small>`
  const slider = $<HTMLInputElement>('wrStep')
  slider.max = String(N)
  slider.value = String(t)
  $('wrPlay').textContent = timer ? '❚❚ Pause' : t >= N ? '↺ Again' : '▶ Play'
}
function wrGo(to: number): void {
  if (!dec) return
  t = Math.max(0, Math.min(dec.steps.length, to))
  if (t >= dec.steps.length) stop()
  wrShow()
}
function stop(): void {
  clearInterval(timer)
  timer = 0
}
$('wrPlay').onclick = () => {
  if (timer) return void (stop(), wrShow())
  if (t >= dec!.steps.length) t = 0
  timer = window.setInterval(() => wrGo(t + 1), calm ? 600 : 260)
  wrGo(t + 1)
}
$('wrNext').onclick = () => (stop(), wrGo(t + 1))
$('wrBack').onclick = () => (stop(), wrGo(t - 1))
$('wrReset').onclick = () => (stop(), wrGo(0))
$<HTMLInputElement>('wrStep').oninput = (e) => (stop(), wrGo(+(e.target as HTMLInputElement).value))

window.addEventListener('resize', () => dec && tfLink())
fetch('./how/decode.json').then((r) => r.json()).then((d: Decode) => {
  dec = d
  document.querySelectorAll<HTMLImageElement>('img.tf-src').forEach((im) => (im.src = d.photo))
  looks = d.steps.map((st) => Uint8Array.from(atob(st.look), (ch) => ch.charCodeAt(0)))
  // the image tokens: the photo cut into its grid of patches, one tile per token
  const [gh, gw] = d.grid
  $('tfImg').style.gridTemplateColumns = `repeat(${gw}, 1fr)`
  $('tfImg').innerHTML = Array.from({ length: gh * gw }, (_, n) =>
    `<i style="background-image:url(${d.photo});background-position:${((n % gw) / (gw - 1)) * 100}% ${(Math.floor(n / gw) / (gh - 1)) * 100}%;background-size:${gw * 100}% ${gh * 100}%"></i>`).join('')
  tfShow()
  if (!calm) setTimeout(tfTick, 850)
  wrShow()
})

// 6 · the real training record of the served model
async function chart(): Promise<void> {
  const t: { loss: [number, number][]; val: [number, number, number][]; images_per_step: number } =
    await (await fetch('./how/training.json')).json()
  const W = 360, H = 220, L = 48, R = 34, T = 14, B = 30
  const xmax = t.loss[t.loss.length - 1][0], lmax = Math.max(...t.loss.map((p) => p[1]))
  const X = (s: number) => L + (s / xmax) * (W - L - R)
  const YL = (v: number) => T + (1 - Math.sqrt(v / lmax)) * (H - T - B) // square root: the long tail stays visible
  const YA = (v: number) => T + (1 - v) * (H - T - B)
  let g = ''
  for (const f of [0, 0.5, 1]) g += `<line x1="${L}" x2="${W - R}" y1="${YA(f)}" y2="${YA(f)}" class="grid"/>
    <text x="${W - R + 4}" y="${YA(f) + 4}" class="acc">${Math.round(100 * f)}%</text>`
  g += `<path d="${t.loss.map(([s, v], i) => `${i ? 'L' : 'M'}${X(s).toFixed(1)},${YL(v).toFixed(1)}`).join('')}" class="loss"/>`
  const best = t.val.reduce((b, v) => (v[1] > b[1] ? v : b))
  const said = new Set([t.val[0], best, t.val[t.val.length - 1]]) // labels on the first, the best and the last
  g += t.val.map((v) => `<circle cx="${X(v[0])}" cy="${YA(v[1])}" r="${t.val.length > 12 ? 3.5 : 5}" class="exact"/>` +
    (said.has(v) ? `<text x="${X(v[0])}" y="${YA(v[1]) - 9}" class="acc" text-anchor="middle">${Math.round(100 * v[1])}%</text>` : '')).join('')
  const imgs = (s: number) => (s * t.images_per_step).toLocaleString()
  g += `<text x="${L}" y="${H - 8}" class="ax">0</text><text x="${W - R}" y="${H - 8}" class="ax" text-anchor="end">${imgs(xmax)} photos</text>
        <text x="${L - 6}" y="${T + 4}" class="lossl" text-anchor="end">wrong</text>`
  $('chart').innerHTML = g
  const first = t.val[0]
  $('chartcap').textContent = `unseen kolams read exactly: ${Math.round(100 * first[1])}% after ${imgs(first[0])} photos, ` +
    `${Math.round(100 * best[1])}% at best, after ${imgs(best[0])}`
}
chart()

function drawAll(): void {
  drawGates()
  render()
  wrBoard.invalidateBase()
  wrShow()
  tfBoard.invalidateBase()
  if (dec) tfShow()
}
let resize = 0
window.addEventListener('resize', () => {
  clearTimeout(resize)
  resize = window.setTimeout(() => {
    gateBoards.forEach((b) => b.invalidateBase())
    play.invalidateBase()
    drawAll()
  }, 150)
})
document.fonts.ready.then(drawAll)
drawAll()

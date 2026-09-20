(function () {
  const canvas = document.getElementById('kolam-logo');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = 440, H = 440;

  // ── Seeded rand ──
  function makeRand(seed) {
    let s = ((seed >>> 0) || 1);
    return () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; };
  }

  // ── Exact sequence from the encoder screenshot ──
  // Single closed stroke on a 7×7 lattice (3×3 anchor grid)
  const SEQ = [
    [2,1],[1,0],[0,1],[1,2],[2,2],[3,2],[4,2],[5,2],[6,1],
    [5,0],[4,1],[4,2],[4,3],[4,4],[4,5],[5,6],[6,5],[5,4],
    [4,4],[3,4],[2,4],[1,4],[0,5],[1,6],[2,5],[2,4],[2,3],
    [2,2],[2,1]
  ];

  // ── Lattice → canvas ──
  // 7×7 lattice, fit with padding
  const PAD = 50;
  const SP = Math.floor(Math.min((W - 2*PAD) / 6, (H - 2*PAD) / 6));
  const OX = Math.round((W - 6*SP) / 2);
  const OY = Math.round((H - 6*SP) / 2);

  function lp(li, lj) {
    return { x: OX + lj*SP, y: OY + li*SP };
  }

  function isAnchor(li, lj) { return li%2===1 && lj%2===1; }

  // ── Spline builder — mirrors decoder.ts buildSplinePts ──
  function buildSplinePts(coords) {
    const n = coords.length;
    if (n < 2) return [];
    const segs = [];
    for (let i = 1; i < n - 1; i++) {
      const mx = (coords[i].x + coords[i+1].x) / 2;
      const my = (coords[i].y + coords[i+1].y) / 2;
      segs.push({
        p0: i === 1 ? coords[0] : segs[segs.length-1].p1,
        cp: coords[i],
        p1: { x: mx, y: my }
      });
    }
    // close back to start
    segs.push({ p0: segs[segs.length-1].p1, cp: coords[n-1], p1: coords[0] });
    const pts = [];
    for (const {p0, cp, p1} of segs) {
      const dx = p1.x-p0.x, dy = p1.y-p0.y;
      const steps = Math.max(8, Math.ceil(Math.sqrt(dx*dx+dy*dy)/2));
      for (let t = 0; t <= 1; t += 1/steps)
        pts.push({
          x: (1-t)*(1-t)*p0.x + 2*(1-t)*t*cp.x + t*t*p1.x,
          y: (1-t)*(1-t)*p0.y + 2*(1-t)*t*cp.y + t*t*p1.y,
        });
    }
    return pts;
  }

  // Pre-build the stroke
  const STROKE = buildSplinePts(SEQ.map(([li,lj]) => lp(li,lj)));

  // ── Pre-render static base (bg + lattice dots) ──
  const base = document.createElement('canvas');
  base.width = W; base.height = H;
  const bctx = base.getContext('2d');
  (function () {
    bctx.fillStyle = '#0B1F1B';
    bctx.fillRect(0, 0, W, H);
    // board texture
    const rand = makeRand(999);
    for (let i = 0; i < Math.floor((W*H)/6000); i++) {
      bctx.beginPath();
      bctx.arc(rand()*W, rand()*H, 0.5+rand()*1.5, 0, Math.PI*2);
      bctx.fillStyle = `rgba(255,255,255,${0.02+rand()*0.06})`;
      bctx.fill();
    }
    // full 7×7 lattice
    for (let li = 0; li < 7; li++) {
      for (let lj = 0; lj < 7; lj++) {
        const {x, y} = lp(li, lj);
        if (isAnchor(li, lj)) {
          bctx.beginPath(); bctx.arc(x, y, 5.5, 0, Math.PI*2);
          bctx.fillStyle = 'rgba(249,250,251,0.88)'; bctx.fill();
        } else {
          bctx.beginPath(); bctx.arc(x, y, 1.8, 0, Math.PI*2);
          bctx.fillStyle = 'rgba(156,163,175,0.4)'; bctx.fill();
        }
      }
    }
  })();

  // ── Chalk dot ──
  function chalkDot(targetCtx, x, y, w, rand) {
    const jx = (rand()-0.5)*w*0.45;
    const jy = (rand()-0.5)*w*0.45;
    const r  = (w/2)*(0.6+rand()*0.6);
    const a  = 0.5+rand()*0.4;
    targetCtx.beginPath();
    targetCtx.arc(x+jx, y+jy, Math.max(r,0.4), 0, Math.PI*2);
    targetCtx.fillStyle = `rgba(249,250,251,${a})`;
    targetCtx.fill();
  }

  // ── Animation state ──
  const STEP  = 4;   // pts per frame
  const TRAIL = 70;  // bright trail length
  const HOLD  = 90;  // frames to hold completed kolam
  const FADE  = 40;  // frames to fade out before restart

  let head = 0, phase = 'drawing', paused = 0;

  // offscreen accumulates the drawn chalk
  const off = document.createElement('canvas');
  off.width = W; off.height = H;
  const offCtx = off.getContext('2d');
  const drawRand = makeRand(4242);

  function reset() {
    head = 0; phase = 'drawing'; paused = 0;
    offCtx.clearRect(0, 0, W, H);
  }
  reset();

  function frame() {
    ctx.drawImage(base, 0, 0);

    if (phase === 'drawing') {
      head = Math.min(head + STEP, STROKE.length);

      // stamp chalk up to head onto offscreen
      offCtx.clearRect(0, 0, W, H);
      const r0 = makeRand(4242);
      for (const {x,y} of STROKE.slice(0, head)) chalkDot(offCtx, x, y, 4, r0);
      ctx.drawImage(off, 0, 0);

      // bright trail on top
      const trail = STROKE.slice(Math.max(0, head-TRAIL), head);
      if (trail.length > 1) {
        const r1 = makeRand(8888 + head);
        for (const {x,y} of trail) chalkDot(ctx, x, y, 6, r1);
      }

      // yellow head dot
      if (head > 0) {
        const tip = STROKE[head-1];
        ctx.beginPath(); ctx.arc(tip.x, tip.y, 6, 0, Math.PI*2);
        ctx.fillStyle = 'rgba(250,204,21,0.95)'; ctx.fill();
      }

      if (head >= STROKE.length) { phase = 'hold'; paused = HOLD; }

    } else if (phase === 'hold') {
      ctx.drawImage(off, 0, 0);
      if (--paused <= 0) { phase = 'fadeout'; paused = FADE; }

    } else if (phase === 'fadeout') {
      ctx.drawImage(off, 0, 0);
      ctx.fillStyle = `rgba(11,31,27,${1 - paused/FADE})`;
      ctx.fillRect(0, 0, W, H);
      if (--paused <= 0) reset();
    }

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
})();

/* ScoreStats — main.js
   Shared utilities loaded on every page.
   ================================================ */

// ── Mobile nav ──────────────────────────────────────────────────
(function () {
  const btn   = document.getElementById('hamburger-btn');
  const links = document.getElementById('nav-links');
  if (btn && links) {
    btn.addEventListener('click', () => links.classList.toggle('open'));
    // Close on outside click
    document.addEventListener('click', (e) => {
      if (!btn.contains(e.target) && !links.contains(e.target)) {
        links.classList.remove('open');
      }
    });
  }
})();

// ── Active nav link highlight ───────────────────────────────────
(function () {
  const path = window.location.pathname;
  document.querySelectorAll('.nav-links a').forEach(a => {
    const href = a.getAttribute('href');
    if (href && href !== '/' && path.startsWith(href)) {
      a.classList.add('active');
    } else if (href === '/' && path === '/') {
      a.classList.add('active');
    }
  });
})();

// ── Auto-dismiss flash messages ─────────────────────────────────
(function () {
  document.querySelectorAll('.flash-msg').forEach(el => {
    setTimeout(() => {
      el.style.transition = 'opacity 0.5s';
      el.style.opacity = '0';
      setTimeout(() => el.remove(), 500);
    }, 4000);
  });
})();

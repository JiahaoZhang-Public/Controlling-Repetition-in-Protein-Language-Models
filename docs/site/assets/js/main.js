// UCCS homepage interactivity
// - result tabs (ESM-3 / ProtGPT2)
// - gallery lightbox
// - BibTeX copy-to-clipboard
// - auto-update "last updated" timestamp
// No framework — vanilla ES2020.

(() => {
  'use strict';

  // -------------------------------------------------------------
  // Tabs
  // -------------------------------------------------------------
  const tabs = document.querySelectorAll('.tab');
  const panels = document.querySelectorAll('.tab-panel');

  function activateTab(target) {
    tabs.forEach((t) => {
      const active = t.dataset.target === target;
      t.classList.toggle('active', active);
      t.setAttribute('aria-selected', active ? 'true' : 'false');
    });
    panels.forEach((p) => {
      const active = p.id === target;
      p.classList.toggle('active', active);
      if (active) {
        p.removeAttribute('hidden');
      } else {
        p.setAttribute('hidden', '');
      }
    });
  }

  tabs.forEach((t) => {
    t.addEventListener('click', () => activateTab(t.dataset.target));
    t.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        e.preventDefault();
        const list = Array.from(tabs);
        const i = list.indexOf(t);
        const next = e.key === 'ArrowRight' ? (i + 1) % list.length : (i - 1 + list.length) % list.length;
        list[next].focus();
        activateTab(list[next].dataset.target);
      }
    });
  });

  // -------------------------------------------------------------
  // Lightbox
  // -------------------------------------------------------------
  const lightbox = document.getElementById('lightbox');
  const lightboxImg = document.getElementById('lightbox-img');
  const lightboxCap = document.getElementById('lightbox-caption');
  const lightboxClose = document.querySelector('.lightbox-close');

  function openLightbox(src, caption) {
    if (!lightbox) return;
    lightboxImg.src = src;
    lightboxImg.alt = caption || '';
    lightboxCap.textContent = caption || '';
    lightbox.removeAttribute('hidden');
    document.body.style.overflow = 'hidden';
  }

  function closeLightbox() {
    if (!lightbox) return;
    lightbox.setAttribute('hidden', '');
    lightboxImg.src = '';
    document.body.style.overflow = '';
  }

  document.querySelectorAll('.gallery-item').forEach((item) => {
    item.addEventListener('click', () => {
      const src = item.dataset.src || item.querySelector('img')?.src;
      const cap = item.dataset.caption || item.querySelector('figcaption')?.textContent;
      if (src) openLightbox(src, cap);
    });
  });

  if (lightboxClose) lightboxClose.addEventListener('click', closeLightbox);
  if (lightbox) {
    lightbox.addEventListener('click', (e) => {
      if (e.target === lightbox) closeLightbox();
    });
  }
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && lightbox && !lightbox.hasAttribute('hidden')) closeLightbox();
  });

  // -------------------------------------------------------------
  // BibTeX copy
  // -------------------------------------------------------------
  document.querySelectorAll('.copy-btn').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const targetId = btn.dataset.copyTarget;
      const target = document.getElementById(targetId);
      if (!target) return;
      const text = target.textContent.trim();
      try {
        await navigator.clipboard.writeText(text);
        const original = btn.textContent;
        btn.textContent = '✓ Copied';
        btn.classList.add('copied');
        setTimeout(() => {
          btn.textContent = original;
          btn.classList.remove('copied');
        }, 1600);
      } catch (err) {
        const range = document.createRange();
        range.selectNode(target);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
      }
    });
  });

  // -------------------------------------------------------------
  // Last-updated stamp — reflect today if the page is loaded later
  // -------------------------------------------------------------
  const stamp = document.getElementById('last-updated');
  if (stamp) {
    const today = new Date().toISOString().slice(0, 10);
    const baked = stamp.textContent.trim();
    if (today > baked) stamp.textContent = today;
  }
})();

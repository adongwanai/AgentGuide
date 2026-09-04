(function () {
  'use strict';

  let mermaidModule = null;
  let rendering = false;

  async function getMermaid() {
    if (mermaidModule) return mermaidModule;
    const mod = await import('https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs');
    mermaidModule = mod.default || mod;
    return mermaidModule;
  }

  function collectBlocks() {
    return Array.from(document.querySelectorAll('pre > code.language-mermaid, pre > code[class*="language-mermaid"]'));
  }

  function normalizeSource(source) {
    return String(source || '')
      .replace(/\r\n?/g, '\n')
      .trim();
  }

  async function renderMermaidBlocks() {
    if (rendering) return;
    rendering = true;

    try {
      const mermaid = await getMermaid();
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: 'loose',
        theme: 'dark',
      });

      const blocks = collectBlocks();
      for (const code of blocks) {
        const pre = code.parentElement;
        if (!pre || pre.dataset.mermaidRendered === 'true') continue;

        const source = normalizeSource(code.textContent);
        if (!source) continue;

        const container = document.createElement('div');
        container.className = 'mermaid-diagram';

        const host = document.createElement('div');
        host.className = 'mermaid';
        host.textContent = source;
        container.appendChild(host);

        pre.insertAdjacentElement('afterend', container);

        try {
          await mermaid.run({ nodes: [host] });
          pre.style.display = 'none';
          pre.dataset.mermaidRendered = 'true';
        } catch (err) {
          container.remove();
          pre.style.display = '';
          pre.dataset.mermaidRendered = 'false';
          console.warn('[mermaid] render failed', err);
        }
      }
    } catch (error) {
      console.error('[mermaid] init failed', error);
    } finally {
      rendering = false;
    }
  }

  document.addEventListener('DOMContentLoaded', renderMermaidBlocks);
  document.addEventListener('astro:after-swap', renderMermaidBlocks);
})();

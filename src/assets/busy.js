/* One explicit request lifecycle for native and Gradio buttons. */
(() => {
  window.RouteSculptorBusy = {
    start(id, label) {
      const button = document.getElementById(id);
      if (!button) return;
      button.dataset.busyLabel = label;
      button.classList.add('is-busy');
      button.setAttribute('aria-busy', 'true');
      button.setAttribute('aria-label', label);
    },
    finish(id) {
      const button = document.getElementById(id);
      if (!button) return;
      button.classList.remove('is-busy');
      button.removeAttribute('aria-busy');
      button.removeAttribute('aria-label');
      delete button.dataset.busyLabel;
    }
  };
})();

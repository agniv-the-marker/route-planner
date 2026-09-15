/* One explicit request lifecycle for native and Gradio buttons. */
(() => {
  function mark(button, label) {
    if (!button) return;
    button.dataset.busyLabel = label;
    button.classList.add('is-busy');
    button.setAttribute('aria-busy', 'true');
    button.setAttribute('aria-label', label);
  }
  function clear(button) {
    if (!button) return;
    button.classList.remove('is-busy');
    button.removeAttribute('aria-busy');
    button.removeAttribute('aria-label');
    delete button.dataset.busyLabel;
  }
  window.RouteSculptorBusy = {
    mark,
    clear,
    start(id, label) {
      mark(document.getElementById(id), label);
    },
    finish(id) {
      clear(document.getElementById(id));
    },
    // Map actions live in server-rendered markup that is usually replaced by the
    // response, so their busy state normally disappears with the node. This clears
    // any button that outlived its request — an unchanged map, or a failed one.
    finishAction(action) {
      document.querySelectorAll(`[data-main-action="${action}"].is-busy`).forEach(clear);
    }
  };
})();

"""Optional pipeline guide, with a native modal dialog on small screens."""

import gradio as gr

PANEL_JS = """
const panel = element.querySelector('#pipeline-panel');
const toggle = element.querySelector('#pipeline-toggle');
const close = element.querySelector('#pipeline-close');
const compact = window.matchMedia('(max-width: 900px)');

function openPanel() {
    panel.dataset.layout = compact.matches ? 'overlay' : 'sidebar';
    if (compact.matches) panel.showModal();
    else panel.show();
    toggle.setAttribute('aria-expanded', 'true');
}

toggle.addEventListener('click', () => {
    if (panel.open) panel.close();
    else openPanel();
});
close.addEventListener('click', () => panel.close());
panel.addEventListener('close', () => {
    // Changing layout closes and immediately reopens the same dialog.
    if (!panel.open) {
        toggle.setAttribute('aria-expanded', 'false');
        toggle.focus({preventScroll: true});
    }
});
panel.addEventListener('click', (event) => {
    if (event.target !== panel || !compact.matches) return;
    const bounds = panel.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right ||
        event.clientY < bounds.top || event.clientY > bounds.bottom) panel.close();
});
element.closest('.gradio-container').addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && panel.open) {
        event.preventDefault();
        panel.close();
    }
});
compact.addEventListener('change', () => {
    if (!panel.open) return;
    panel.close();
    openPanel();
});
"""


def app_header(connected: bool) -> gr.HTML:
    status = (
        "Inspection is ready. Models load when you start."
        if connected
        else "Preview mode: add images to see the flow. Inspection needs a model worker."
    )
    pipeline_label = "Ready" if connected else "Preview mode"
    return gr.HTML(
        f"""
        <header class="app-header">
          <div class="header-copy">
            <h1>Visual Peer Inspector</h1>
            <p>Compare objects across multiple views of the same scene.</p>
          </div>
          <div class="header-actions">
            <span class="connection-status">{pipeline_label}</span>
            <button id="pipeline-toggle" type="button" aria-controls="pipeline-panel"
                    aria-expanded="false">How it works</button>
          </div>
        </header>
        <dialog id="pipeline-panel" aria-labelledby="pipeline-title">
          <div class="pipeline-heading">
            <h2 id="pipeline-title">How it works</h2>
            <button id="pipeline-close" type="button" autofocus>Close</button>
          </div>
          <p class="pipeline-intro">{status}</p>
          <ol class="pipeline-steps">
            <li><h3>Photograph one group</h3><p>Add 2–5 views of the same arrangement.</p></li>
            <li><h3>Inspect the views</h3><p>The app locates objects and combines evidence into one ranking.</p></li>
            <li><h3>Check the matches</h3><p>Review which boxes show the same physical object. Assisted mode lets you confirm these links.</p></li>
            <li><h3>Record and save</h3><p>Make your own physical finding, then download a case ZIP and report.</p></li>
          </ol>
          <p class="pipeline-note">The vision stages use LocateAnything-3B, SAM 2.1 and DINOv2-small.
            Unusual does not always mean defective. A tie has no unique selection.
            The model ranking and your decision are saved separately.</p>
        </dialog>
        """,
        js_on_load=PANEL_JS,
        apply_default_css=False,
        elem_id="app-heading",
    )

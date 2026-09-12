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
        "Models load when you start an inspection. Run details are included in the report."
        if connected
        else "These models are not connected to this app yet."
    )
    pipeline_label = "Inspection pipeline" if connected else "Planned pipeline"
    return gr.HTML(
        f"""
        <header class="app-header">
          <h1>Visual Peer Inspector</h1>
          <button id="pipeline-toggle" type="button" aria-controls="pipeline-panel"
                  aria-expanded="false">How it works</button>
        </header>
        <dialog id="pipeline-panel" aria-labelledby="pipeline-title">
          <div class="pipeline-heading">
            <h2 id="pipeline-title">How it works</h2>
            <button id="pipeline-close" type="button" autofocus>Close</button>
          </div>
          <p class="pipeline-status">{pipeline_label}</p>
          <p class="pipeline-intro">{status}</p>
          <ol class="pipeline-steps">
            <li><h3>Find objects</h3>
              <p>Locate each object in every image.</p>
              <span>LocateAnything-3B</span></li>
            <li><h3>Create masks</h3>
              <p>Separate each object from its background.</p>
              <span>SAM 2.1 Hiera-large</span></li>
            <li><h3>Extract features</h3>
              <p>Represent the appearance of each object.</p>
              <span>DINOv2-small</span></li>
            <li><h3>Compare peers</h3>
              <p>Rank objects by how different they look from similar objects.</p>
              <span>P1 patch comparison</span></li>
            <li><h3>Combine views</h3>
              <p>Match the same object across images and combine its evidence.</p>
              <span>Cross-view matching and rank fusion</span></li>
          </ol>
          <p class="pipeline-note">Unusual does not always mean defective.
            Uncertain object matches may need your review.</p>
        </dialog>
        """,
        js_on_load=PANEL_JS,
        apply_default_css=False,
        elem_id="app-heading",
    )

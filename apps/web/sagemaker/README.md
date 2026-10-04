# Private SageMaker Studio demo

Use the application source package with Python 3.12, a T4 GPU and 50 GB
persistent storage. The tested instance is `ml.g4dn.2xlarge`.
Dataset archives and experiment feature caches are unnecessary for fresh uploads.

## First install

Extract the source package into `/home/sagemaker-user/visual-peer-inspector`.
Run from that folder:

```bash
bash apps/web/sagemaker/setup_sagemaker.sh
VPI_DINO_PYTHON="$PWD/.runtime/dino/bin/python" \
  .runtime/inference/bin/python apps/web/sagemaker/prepare_models.py
python apps/web/sagemaker/service.py start
```

Setup validates the package manifest, isolates three dependency environments,
checks pinned SAM source and CUDA, and fails on dependency conflicts. The single
decord wheel-tag exception requires the known Linux wheel and a loaded native
library; video decoding is outside this RGB app's scope.
Model preparation verifies pinned assets and exercises each model. Starting the
server only verifies HTTP availability; it does not establish upload accuracy.

Save the app's complete **HTTPS** Studio proxy URL, without a trailing slash,
in `.runtime/studio-app-url.txt` before starting the server. This private local
configuration persists through restart and is excluded from the source package.
Gradio uses it to produce HTTPS download links when Studio's proxy does not
forward the original protocol. Do not disable browser download protection.

## Restart an installed Space

After starting the Space, open its JupyterLab terminal:

```bash
cd /home/sagemaker-user/visual-peer-inspector
python apps/web/sagemaker/service.py start
python apps/web/sagemaker/service.py status
```

The helper preserves an existing owned server, refuses to replace a foreign
listener, and checks process identity before stopping it. To stop an idle app:

```bash
python apps/web/sagemaker/service.py stop
```

Cancel any running inspection in the app first. Logs and the PID stay in `logs/`.
Keep the Space and app server running for the demo. Closing a browser tab does
not stop them. Space shutdown stops the app; automatic Space startup is not
configured by these helpers.

## Open the interface

Append `/proxy/7860/` to the Space's `/jupyterlab/default` URL. The app binds to
loopback, uses Studio's authenticated proxy, and does not create a public share
link. Its URL requires AWS access to the private Space.

Automatic matching remains reviewable. Assisted matching confirms physical
identities before the frozen scorer ranks objects. A saved case preserves the
model result separately from the operator's finding. Test the intended photos
before presenting a demonstration; the score is not a correctness probability.

## Optional fixed free ngrok address

This route forwards only the app on port 7860. It requires the account's assigned
free HTTPS development domain and the official Linux ngrok agent installed at
`.runtime/ngrok/ngrok`. Keep its minimal token configuration in
`.runtime/ngrok/ngrok.yml` with mode 600. Keep `.runtime/ngrok` owner-only and
outside Git. Do not upload a token or demo password in the application package.

The account owner enters the separate demo password at a hidden prompt:

```bash
python apps/web/sagemaker/configure_demo.py \
  --url https://YOUR-ASSIGNED-DOMAIN.ngrok-free.dev --username demo
```

Once an inspection is saved and the app is idle, switch to protected mode:

```bash
python apps/web/sagemaker/service.py stop
python apps/web/sagemaker/fixed_demo.py start-app --settings .runtime/ngrok/demo.json
nohup python apps/web/sagemaker/fixed_demo.py start-tunnel \
  --settings .runtime/ngrok/demo.json > logs/ngrok.log 2>&1 < /dev/null &
```

The helper refuses public forwarding if the app accepts anonymous configuration
requests, the configured login fails, or Gradio's download root differs from the
fixed URL. Request inspection is disabled. Verify the external login, upload and
download flow before sharing the address. The local authentication check alone
does not verify public connectivity.

After a Space restart, run `start-app` and `start-tunnel` again. The same assigned
domain is reused; the Space, app and tunnel must all be running. This is a
supervised demo subject to the free plan's limits, not always-on hosting. Do not
start multiple tunnel agents for the same address. Stop the owned ngrok process
before returning to private Studio mode with `service.py stop` / `start`.

References: [ngrok free plan](https://ngrok.com/docs/pricing-limits/free-plan-limits)
and [Gradio authentication](https://www.gradio.app/guides/sharing-your-app).

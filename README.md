# Purrception

An interactive webcam prototype that trains a small expression classifier in the browser, then maps its predictions to local meme artwork.

## Run it

```bash
npm install
npm run dev
```

Open the local URL printed by Vite. Camera access requires `localhost`/`127.0.0.1` or HTTPS. Allow camera access when prompted.

## Deploy to GitHub Pages

Push the `main` branch to GitHub. The workflow in [`.github/workflows/deploy.yml`](./.github/workflows/deploy.yml) builds and deploys the app automatically to `https://mleu8.github.io/meme/`.

In the repository settings, open **Pages** and set **Source** to **GitHub Actions**. The production build uses `/meme/` as its base path so scripts and local meme artwork load from the project-page URL.

## How it works

- Participants first review the image/model-training consent and must check an unchecked box before continuing. The camera is not requested until they finish the training instructions and choose **I’m ready — start training**. Declining exits without starting the camera.
- **Training Booth** uses ml5 FaceMesh to extract normalized facial-expression measurements during the final two seconds of each training countdown. One separate still photo is captured at the end for the training strip; the saved photo pixels are not classifier inputs.
- Each participant contributes labeled FaceMesh measurements for four selected emotions. The local classifier uses accumulated measurements for the current participant’s selected labels; it is not a pretrained emotion detector.
- Starting another participant clears the previous participant’s photos and results but keeps the shared landmark training measurements in page memory. New measurements are added to that set. Four selected emotions must each contribute valid FaceMesh samples before the Meme Booth unlocks.
- **Meme Booth** continuously classifies live facial-expression measurements and stabilizes predictions over a short rolling window. A matching local cat meme remains visible until the stable prediction changes; participants choose when to capture each of four photo/meme pairs. Predictions interpret visible expressions and may not reflect how a participant actually feels.
- Training photos use the existing three-second countdown; Meme Booth photos are captured immediately when the participant presses **Take Photo**. The finished strips can be downloaded as a PNG.
- Camera frames and captured photos are held in page memory and cleared when starting another participant. The accumulated landmark training measurements stay in page memory across participants; refreshing or closing the page clears them. Downloaded strips are files on the participant’s device and are not deleted by the app.
- This repository contains no application server, upload endpoint, or persistent browser storage for participant data. The app code does not send participant images or landmark measurements to a server. It does load Google Fonts and ml5/MediaPipe dependencies; hosting and dependency-provider access/log retention is not established by this repository and should be confirmed before public participant use.
- The labels describe the expression being performed for the camera, not the participant’s actual feelings.

## Replace the meme images

The placeholder artwork lives in [`public/memes/`](./public/memes/). Each category has two clearly named SVGs:

- `sad-blue.svg`, `sad-rain.svg`
- `happy-sun.svg`, `happy-party.svg`
- `excited-wow.svg`, `excited-stars.svg`
- `scared-spooky.svg`, `scared-gasp.svg`

Replace any file with a JPG, PNG, WebP, or SVG while keeping the same filename, or update the paths in the `MEMES` object near the top of [`src/main.jsx`](./src/main.jsx). Keep the images landscape-oriented for the best fit in the strip. The app rotates between the two available local options as each meme-booth photo is captured.

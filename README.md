# Face to Meme Lab

An interactive webcam prototype that trains a small expression classifier in the browser, then maps its predictions to local meme artwork.

## Run it

```bash
npm install
npm run dev
```

Open the local URL printed by Vite. Camera access requires `localhost`/`127.0.0.1` or HTTPS. Allow camera access when prompted.

## How it works

- **Training Booth** uses MobileNet as a visual feature extractor. The classifier on top is trained in TensorFlow.js from the participant’s four labeled webcam photos; it is not a pretrained emotion detector.
- Each training participant contributes one happy, sad, excited, and scared photo. The photos are printed into a one-column by four-row strip.
- Samples accumulate across participants in the current page session, so additional people can contribute to the same classifier. The model requires at least four examples per expression before training is enabled, so the first few participants may be asked to add more examples.
- **Meme Booth** captures four unprompted photos and classifies each captured photo once when a trained model is available. The selected meme and predicted category are stored with that row and do not change while the strip is displayed. If there is not enough training data yet, it still prints local preview memes for every row and labels them as previews rather than pretending they are model predictions.
- Both booths use a three-second countdown and a camera-flash animation. The finished strips can be downloaded as a PNG.
- Camera frames, feature vectors, and the trained model stay in browser memory. They are not uploaded. Refreshing or closing the page removes the samples and model. The **Delete samples & reset model** control removes them immediately.
- Predictions are smoothed over the most recent frames. A low-confidence result shows **Try another expression** rather than choosing a meme.
- The labels describe the expression being performed for the camera, not the participant’s actual feelings.

## Replace the meme images

The placeholder artwork lives in [`public/memes/`](./public/memes/). Each category has two clearly named SVGs:

- `sad-blue.svg`, `sad-rain.svg`
- `happy-sun.svg`, `happy-party.svg`
- `excited-wow.svg`, `excited-stars.svg`
- `scared-spooky.svg`, `scared-gasp.svg`

Replace any file with a JPG, PNG, WebP, or SVG while keeping the same filename, or update the paths in the `MEMES` object near the top of [`src/main.jsx`](./src/main.jsx). Keep the images landscape-oriented for the best fit in the strip. The app rotates between the two available local options as each meme-booth photo is captured.

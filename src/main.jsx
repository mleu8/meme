import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';
import {
  ArrowLeft, Camera, Check, ChevronRight, CircleHelp, Download, Heart,
  LockKeyhole, RefreshCw, Sparkles, Trash2, TriangleAlert, Users
} from 'lucide-react';
import './styles.css';

const LABELS = ['happy', 'sad', 'excited', 'scared'];
const LABEL_META = {
  happy: { prompt: 'Show us your happy face!', color: '#efb83e', emoji: '☺' },
  sad: { prompt: 'Show us your sad face!', color: '#6e94d4', emoji: '☹' },
  excited: { prompt: 'Show us your excited face!', color: '#e77862', emoji: '✦' },
  scared: { prompt: 'Show us your scared face!', color: '#9275cc', emoji: '◉' }
};
const ASSET_BASE = import.meta.env.BASE_URL;
const MEMES = {
  happy: [`${ASSET_BASE}memes/happy-sun.svg`, `${ASSET_BASE}memes/happy-party.svg`],
  sad: [`${ASSET_BASE}memes/sad-blue.svg`, `${ASSET_BASE}memes/sad-rain.svg`],
  excited: [`${ASSET_BASE}memes/excited-wow.svg`, `${ASSET_BASE}memes/excited-stars.svg`],
  scared: [`${ASSET_BASE}memes/scared-spooky.svg`, `${ASSET_BASE}memes/scared-gasp.svg`]
};
const PLACEHOLDER_MEME = `${ASSET_BASE}memes/placeholder.svg`;
const TRAINING_MINIMUM = 4;

const emptySamples = () => Object.fromEntries(LABELS.map(label => [label, []]));

function App() {
  const [booth, setBooth] = useState('training');
  const [cameraState, setCameraState] = useState('idle');
  const [cameraError, setCameraError] = useState('');
  const [countdown, setCountdown] = useState(null);
  const [flash, setFlash] = useState(false);
  const [captureBusy, setCaptureBusy] = useState(false);
  const [trainingPhotos, setTrainingPhotos] = useState([]);
  const [trainingIndex, setTrainingIndex] = useState(0);
  const [trainingComplete, setTrainingComplete] = useState(false);
  const [memePhotos, setMemePhotos] = useState([]);
  const [memeIndex, setMemeIndex] = useState(0);
  const [memeComplete, setMemeComplete] = useState(false);
  const [samples, setSamples] = useState(emptySamples);
  const [modelStatus, setModelStatus] = useState('empty');
  const [modelProgress, setModelProgress] = useState(0);
  const [notice, setNotice] = useState('');
  const [memeNotice, setMemeNotice] = useState('');
  const [stripVariant, setStripVariant] = useState(0);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const featureModelRef = useRef(null);
  const classifierRef = useRef(null);
  const countdownTimerRef = useRef(null);
  const memePhotosRef = useRef([]);
  const memeCountRef = useRef(0);

  const totalSamples = useMemo(() => LABELS.reduce((total, label) => total + samples[label].length, 0), [samples]);
  const activeTrainLabel = LABELS[trainingIndex];
  const modelReady = modelStatus === 'ready' && classifierRef.current;
  const canTrain = LABELS.every(label => samples[label].length >= TRAINING_MINIMUM);

  const attachStream = useCallback(async () => {
    if (!streamRef.current || !videoRef.current) return;
    if (videoRef.current.srcObject !== streamRef.current) videoRef.current.srcObject = streamRef.current;
    await videoRef.current.play().catch(() => {});
  }, []);

  const startCamera = useCallback(async () => {
    if (streamRef.current) {
      await attachStream();
      setCameraState('ready');
      return true;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraState('error');
      setCameraError('This browser does not support webcam access. Try the latest Chrome, Safari, or Firefox.');
      return false;
    }
    setCameraState('loading');
    setCameraError('');
    try {
      streamRef.current = await navigator.mediaDevices.getUserMedia({
        video: { width: 720, height: 540, facingMode: 'user' },
        audio: false
      });
      await attachStream();
      setCameraState('ready');
      return true;
    } catch (error) {
      setCameraState('error');
      setCameraError(error.name === 'NotAllowedError'
        ? 'Camera access was blocked. Allow it in your browser settings, then try again.'
        : 'We could not start the camera. Check that another app is not using it.');
      return false;
    }
  }, [attachStream]);

  useEffect(() => {
    startCamera();
    return () => {
      streamRef.current?.getTracks().forEach(track => track.stop());
      clearInterval(countdownTimerRef.current);
    };
  }, [startCamera]);

  useEffect(() => {
    attachStream();
  }, [booth, attachStream]);

  useEffect(() => {
    memePhotosRef.current = memePhotos;
    memeCountRef.current = memePhotos.length;
  }, [memePhotos]);

  const loadFeatureModel = async () => {
    if (!featureModelRef.current) featureModelRef.current = await mobilenet.load({ version: 2, alpha: 1 });
    return featureModelRef.current;
  };

  const captureDataUrl = () => {
    const video = videoRef.current;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 720;
    canvas.height = video.videoHeight || 540;
    const context = canvas.getContext('2d');
    context.translate(canvas.width, 0);
    context.scale(-1, 1);
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.9);
  };

  const runCountdown = useCallback(async (onCapture) => {
    if (captureBusy) return;
    if (!(await startCamera()) || !videoRef.current?.videoWidth) {
      setNotice('The camera is not ready yet. Allow camera access and try again.');
      return;
    }
    setCaptureBusy(true);
    setNotice('');
    setCountdown(3);
    let remaining = 3;
    countdownTimerRef.current = setInterval(() => {
      remaining -= 1;
      if (remaining <= 0) {
        clearInterval(countdownTimerRef.current);
        setCountdown(null);
        setFlash(true);
        window.setTimeout(() => setFlash(false), 180);
        onCapture();
      } else setCountdown(remaining);
    }, 1000);
  }, [captureBusy, startCamera]);

  const captureTrainingPhoto = async () => {
    try {
      const model = await loadFeatureModel();
      const tensor = model.infer(videoRef.current, true);
      const vector = Array.from(await tensor.data());
      tensor.dispose();
      const photo = captureDataUrl();
      setTrainingPhotos(current => [...current, { photo, label: activeTrainLabel }]);
      setSamples(current => ({ ...current, [activeTrainLabel]: [...current[activeTrainLabel], vector] }));
      if (trainingIndex === LABELS.length - 1) setTrainingComplete(true);
      else setTrainingIndex(index => index + 1);
    } catch {
      setNotice('This photo could not be read. Keep your face in frame and try again.');
    } finally {
      setCaptureBusy(false);
    }
  };

  const captureMemePhoto = async () => {
    try {
      let label = null;
      let confidence = null;
      let matched = false;
      if (classifierRef.current) {
        const model = await loadFeatureModel();
        const tensor = model.infer(videoRef.current, true);
        const output = classifierRef.current.predict(tensor);
        const values = Array.from(await output.data());
        const bestIndex = values.indexOf(Math.max(...values));
        confidence = values[bestIndex];
        if (confidence >= 0.58) {
          label = LABELS[bestIndex];
          matched = true;
        }
        output.dispose();
        tensor.dispose();
      }
      const currentPhotos = memePhotosRef.current;
      const photoNumber = memeCountRef.current;
      const previewLabel = LABELS[photoNumber % LABELS.length];
      const imageLabel = label || previewLabel;
      const photo = captureDataUrl();
      const meme = MEMES[imageLabel][photoNumber % 2];
      const nextPhotos = [...currentPhotos, { photo, label: imageLabel, confidence, meme, matched }];
      memePhotosRef.current = nextPhotos;
      memeCountRef.current = nextPhotos.length;
      setMemePhotos(nextPhotos);
      if (!matched) setMemeNotice('Showing a local preview meme for this photo. Train the model to get a personalized match.');
      if (nextPhotos.length >= 4) {
        setMemeComplete(true);
        setNotice('Your meme strip is ready!');
      }
      else setMemeIndex(index => index + 1);
    } catch {
      setMemeNotice('That photo could not be collected. Keep the camera preview visible and try again.');
      setNotice('The photo was not added, so this is still photo 1 of 4.');
    } finally {
      setCaptureBusy(false);
    }
  };

  const trainClassifier = async () => {
    if (!canTrain) return;
    setModelStatus('training');
    setModelProgress(5);
    try {
      const features = [];
      const labels = [];
      LABELS.forEach((label, labelIndex) => samples[label].forEach(vector => {
        features.push(vector);
        labels.push(labelIndex);
      }));
      const x = tf.tensor2d(features);
      const y = tf.oneHot(tf.tensor1d(labels, 'int32'), LABELS.length);
      const classifier = tf.sequential();
      classifier.add(tf.layers.dense({ inputShape: [features[0].length], units: 64, activation: 'relu' }));
      classifier.add(tf.layers.dropout({ rate: 0.2 }));
      classifier.add(tf.layers.dense({ units: LABELS.length, activation: 'softmax' }));
      classifier.compile({ optimizer: tf.train.adam(0.0008), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
      await classifier.fit(x, y, {
        epochs: 24,
        batchSize: Math.min(16, features.length),
        shuffle: true,
        callbacks: { onEpochEnd: epoch => setModelProgress(Math.round(((epoch + 1) / 24) * 95)) }
      });
      classifierRef.current?.dispose();
      classifierRef.current = classifier;
      x.dispose();
      y.dispose();
      setModelProgress(100);
      setModelStatus('ready');
      setNotice('Your shared expression model is ready for the Meme Booth.');
    } catch {
      setModelStatus('error');
      setNotice('Training did not finish. Capture more examples and try again.');
    }
  };

  const completeTraining = async () => {
    setTrainingComplete(true);
    if (canTrain) await trainClassifier();
  };

  const enterMemeBooth = () => {
    setBooth('meme');
    setMemePhotos([]);
    memePhotosRef.current = [];
    memeCountRef.current = 0;
    setMemeIndex(0);
    setMemeComplete(false);
    setMemeNotice(modelReady ? '' : 'The booth will still print local preview memes. Train the model for personalized matches.');
  };

  const resetModel = () => {
    classifierRef.current?.dispose();
    classifierRef.current = null;
    setSamples(emptySamples());
    setModelStatus('empty');
    setModelProgress(0);
    setNotice('All browser-local training examples and the model were deleted.');
  };

  const newTrainingParticipant = () => {
    setTrainingPhotos([]);
    setTrainingIndex(0);
    setTrainingComplete(false);
    setNotice('');
    setBooth('training');
  };

  const tryAgain = () => {
    setMemePhotos([]);
    memePhotosRef.current = [];
    memeCountRef.current = 0;
    setMemeIndex(0);
    setMemeComplete(false);
    setMemeNotice(modelReady ? '' : 'The booth will still print local preview memes. Train the model for personalized matches.');
    setBooth('meme');
  };

  return (
    <main className={`app-shell ${booth === 'meme' ? 'meme-mode' : ''}`}>
      <header className="topbar">
        <div className="brand"><div className="brand-mark"><Sparkles size={18} /></div><span>face to meme lab</span></div>
        <div className="privacy-chip"><LockKeyhole size={14} /> camera + training stay local</div>
      </header>
      <div className="booth-switcher">
        <button className={booth === 'training' ? 'active' : ''} onClick={() => setBooth('training')}><span>01</span> training booth</button>
        <i />
        <button className={booth === 'meme' ? 'active' : ''} onClick={enterMemeBooth}><span>02</span> meme booth</button>
      </div>
      {booth === 'training' ? (
        <TrainingBooth
          videoRef={videoRef} cameraState={cameraState} cameraError={cameraError} startCamera={startCamera}
          countdown={countdown} flash={flash} captureBusy={captureBusy} activeLabel={activeTrainLabel}
          trainingIndex={trainingIndex} trainingPhotos={trainingPhotos} trainingComplete={trainingComplete}
          runCountdown={() => runCountdown(captureTrainingPhoto)} completeTraining={completeTraining}
          enterMemeBooth={enterMemeBooth} trainClassifier={trainClassifier} modelStatus={modelStatus}
          modelProgress={modelProgress} canTrain={canTrain} totalSamples={totalSamples} notice={notice}
          newTrainingParticipant={newTrainingParticipant} resetModel={resetModel}
        />
      ) : (
        <MemeBooth
          videoRef={videoRef} cameraState={cameraState} cameraError={cameraError} startCamera={startCamera}
          countdown={countdown} flash={flash} captureBusy={captureBusy} memePhotos={memePhotos} memeComplete={memeComplete}
          runCountdown={() => runCountdown(captureMemePhoto)} tryAgain={tryAgain}
          memeNotice={memeNotice} setBooth={setBooth} totalSamples={totalSamples}
        />
      )}
      <footer><span><Users size={14} /> {totalSamples} shared local training examples</span><span>Predictions are interpretations of expressions, not feelings.</span></footer>
    </main>
  );
}

function BoothCamera({ videoRef, cameraState, cameraError, startCamera, countdown, flash, children, theme }) {
  return <div className={`booth-stage ${theme || ''}`}>
    <div className="curtain curtain-left" /><div className="curtain curtain-right" />
    <div className="camera-frame">
      <video ref={videoRef} muted playsInline className="webcam" />
      {cameraState !== 'ready' && <div className="camera-placeholder">
        {cameraState === 'error'
          ? <><TriangleAlert size={28} /><strong>Camera unavailable</strong><p>{cameraError}</p><button className="button cream" onClick={startCamera}>Try camera again</button></>
          : <><Camera size={28} /><strong>{cameraState === 'loading' ? 'Waking up the booth…' : 'Camera preview'}</strong><button className="button cream" onClick={startCamera}>Allow camera</button></>}
      </div>}
      {countdown && <div className="countdown"><span>{countdown}</span><small>hold that pose</small></div>}
      {flash && <div className="flash" />}
      {children}
    </div>
    <div className="camera-base"><span className="live-dot" /> live mirror <span><LockKeyhole size={12} /> never uploaded</span></div>
  </div>;
}

function TrainingBooth(props) {
  const { activeLabel, trainingIndex, trainingPhotos, trainingComplete, runCountdown, completeTraining, enterMemeBooth, trainClassifier, modelStatus, modelProgress, canTrain, totalSamples, notice, newTrainingParticipant, resetModel } = props;
  return <section className="booth-page">
    <div className="booth-intro"><div className="eyebrow"><span className="eyebrow-dot" /> booth one / build the brain</div><h1>Strike a pose.<br /><em>Train the AI.</em></h1><p>Four photos. Four expressions. Your examples join the shared class model.</p></div>
    <div className="booth-layout">
      <div><BoothCamera {...props} theme="training-curtain"><div className="booth-prompt"><span>photo {Math.min(trainingIndex + 1, 4)} of 4</span><strong>{trainingComplete ? 'All poses captured!' : LABEL_META[activeLabel].prompt}</strong><small>{trainingComplete ? 'Print your strip below.' : 'The flash will capture one photo.'}</small></div></BoothCamera>
        <div className="booth-controls">{!trainingComplete ? <><div className="progress-dots">{LABELS.map((label, index) => <i key={label} className={index <= trainingIndex ? 'on' : ''} style={{ '--dot': LABEL_META[label].color }} />)}</div><button className="button yellow wide" onClick={runCountdown} disabled={props.captureBusy || props.countdown}><Camera size={18} /> {props.captureBusy ? 'Printing your photo…' : 'Take this photo'}</button></> : <button className="button yellow wide" onClick={completeTraining}><Sparkles size={17} /> Print my photo strip</button>}</div>
      </div>
      <div className="strip-column">
        {trainingComplete ? <><PhotoStrip photos={trainingPhotos} type="training" /><h2 className="thank-you">Thank you for training!</h2><div className="strip-actions"><button className="button dark" onClick={() => downloadStrip(trainingPhotos, 'training')}><Download size={17} /> Download Photo Strip</button><button className="button cream dark-text" onClick={enterMemeBooth}>Enter the Meme Booth <ChevronRight size={17} /></button></div></> : <EmptyStrip type="training" />}
      </div>
    </div>
    {trainingComplete && <div className="model-panel"><div><b>{canTrain ? 'Model ready to learn' : 'More examples make it smarter'}</b><p>{canTrain ? 'Train the growing model before entering the Meme Booth.' : `This participant added 4 examples. The model currently has ${totalSamples} total; collect at least ${TRAINING_MINIMUM} examples per expression to train reliably.`}</p></div>{canTrain && <button className="button dark" onClick={trainClassifier} disabled={modelStatus === 'training'}>{modelStatus === 'training' ? `Training… ${modelProgress}%` : modelStatus === 'ready' ? 'Retrain shared model' : 'Train shared model'}</button>}</div>}
    {notice && <div className="notice"><Check size={15} /> {notice}</div>}
    <div className="utility-row"><button className="text-button" onClick={newTrainingParticipant}><RefreshCw size={14} /> Start another training participant</button><button className="danger-button" onClick={resetModel}><Trash2 size={14} /> Delete all shared samples</button></div>
  </section>;
}

function MemeBooth(props) {
  const { memePhotos, memeComplete, runCountdown, tryAgain, memeNotice, setBooth, totalSamples } = props;
  const complete = memeComplete || memePhotos.length >= 4;
  return <section className="booth-page">
    <div className="booth-intro"><div className="eyebrow"><span className="eyebrow-dot purple" /> booth two / the reveal</div><h1>Four photos.<br /><em>Any expression.</em></h1><p>What will the AI see? Make any face you want. The model decides.</p></div>
    {!complete ? <div className="booth-layout meme-layout"><div><BoothCamera {...props} theme="meme-curtain"><div className="booth-prompt"><span>photo {memePhotos.length + 1} of 4</span><strong>Make any expression.</strong><small>No emotion category is shown or selected.</small></div></BoothCamera><div className="booth-controls"><button className="button coral wide" onClick={runCountdown} disabled={props.captureBusy || props.countdown}><Camera size={18} /> {props.captureBusy ? 'Reading your expression…' : `Take photo ${memePhotos.length + 1} of 4`}</button></div></div><div className="strip-column"><EmptyStrip type="meme" /><div className="model-note"><Sparkles size={16} /><span><b>Private model interpretation.</b> Each captured photo gets its own fixed meme match.</span></div></div></div> : <div className="completed-meme"><PhotoStrip photos={memePhotos} type="meme" /><h2>Your meme strip is ready!</h2><p>Each meme is the model’s interpretation of that individual expression.</p><div className="strip-actions"><button className="button dark" onClick={() => downloadStrip(memePhotos, 'meme')}><Download size={17} /> Download Meme Strip</button><button className="button cream dark-text" onClick={tryAgain}><RefreshCw size={17} /> Try Again</button></div></div>}
    {memeNotice && <div className="notice warning"><CircleHelp size={15} /> {memeNotice}</div>}
    <button className="back-training" onClick={() => setBooth('training')}><ArrowLeft size={14} /> Back to Training Booth <span>{totalSamples} examples in shared model</span></button>
  </section>;
}

function EmptyStrip({ type }) {
  return <div className={`empty-strip ${type}`}><div className="slot-label">{type === 'training' ? 'your four poses' : 'your four matches'}</div>{Array.from({ length: 4 }, (_, index) => <div className="empty-row" key={index}><span>{index + 1}</span><i>{type === 'training' ? 'photo waiting' : 'meme waiting'}</i></div>)}</div>;
}

function PhotoStrip({ photos, type }) {
  return <div className={`photo-strip ${type} printed`}><div className="strip-top"><Sparkles size={13} /><span>FACE TO MEME LAB</span><Sparkles size={13} /></div>{photos.map((item, index) => <div className="strip-row" key={`${item.photo}-${index}`}><div className="strip-photo"><img src={item.photo} alt={`${type} photo ${index + 1}`} /><small>{index + 1}</small></div>{type === 'meme' && <div className="strip-meme"><img src={item.meme} alt={`${item.label} meme`} /><small>{item.matched ? `model sees: ${item.label}` : `preview image: ${item.label}`}</small></div>}</div>)}<div className="strip-bottom">made locally · <Heart size={10} fill="currentColor" /></div></div>;
}

async function downloadStrip(items, type) {
  const width = type === 'training' ? 360 : 560;
  const rowHeight = 210;
  const height = 72 + rowHeight * 4 + 42;
  const canvas = document.createElement('canvas');
  canvas.width = width * 2;
  canvas.height = height * 2;
  const context = canvas.getContext('2d');
  context.scale(2, 2);
  context.fillStyle = '#fff8e9';
  context.fillRect(0, 0, width, height);
  context.strokeStyle = '#d7c9ae';
  context.lineWidth = 2;
  context.strokeRect(12, 12, width - 24, height - 24);
  context.fillStyle = '#302822';
  context.font = 'bold 12px monospace';
  context.textAlign = 'center';
  context.fillText('✦  FACE TO MEME LAB  ✦', width / 2, 42);
  for (let index = 0; index < items.length; index += 1) {
    const y = 72 + index * rowHeight;
    await drawImageCover(context, items[index].photo, 24, y, type === 'training' ? width - 48 : (width - 72) / 2, 170);
    if (type === 'meme') {
      await drawImageCover(context, items[index].meme, width / 2 + 12, y, (width - 72) / 2, 170);
      context.fillStyle = '#302822';
      context.font = 'bold 11px monospace';
      context.fillText(items[index].matched ? `MODEL: ${items[index].label.toUpperCase()}` : `PREVIEW: ${items[index].label.toUpperCase()}`, width * .75, y + 190);
    }
    context.fillStyle = '#a6987f';
    context.font = '10px monospace';
    context.textAlign = 'left';
    context.fillText(String(index + 1).padStart(2, '0'), 26, y + 190);
  }
  context.textAlign = 'center';
  context.fillStyle = '#8e806b';
  context.font = '10px monospace';
  context.fillText('PRINTED LOCALLY · EXPRESSIONS ARE PLAYFUL PREDICTIONS', width / 2, height - 28);
  const link = document.createElement('a');
  link.download = `${type}-photo-strip.png`;
  link.href = canvas.toDataURL('image/png');
  link.click();
}

function drawImageCover(context, source, x, y, width, height) {
  return new Promise(resolve => {
    const image = new Image();
    image.onload = () => {
      const scale = Math.max(width / image.width, height / image.height);
      const sourceWidth = width / scale;
      const sourceHeight = height / scale;
      const sourceX = (image.width - sourceWidth) / 2;
      const sourceY = (image.height - sourceHeight) / 2;
      context.drawImage(image, sourceX, sourceY, sourceWidth, sourceHeight, x, y, width, height);
      resolve();
    };
    image.onerror = resolve;
    image.src = source;
  });
}

createRoot(document.getElementById('root')).render(<App />);

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import * as mobilenet from '@tensorflow-models/mobilenet';
import {
  Camera, Check, ChevronRight, Download, LockKeyhole, RefreshCw, Sparkles,
  Trash2, TriangleAlert, Users
} from 'lucide-react';
import './styles.css';

const EMOTIONS = ['angry', 'awkward', 'disgusted', 'happy', 'scared', 'serious', 'shocked', 'stressed', 'tired'];
const EMOTION_META = {
  angry: { prompt: 'Give us your angry face', color: '#df765f', emoji: '↯' },
  awkward: { prompt: 'Show us your awkward face', color: '#d49a69', emoji: '⌁' },
  disgusted: { prompt: 'Make your most disgusted face', color: '#9a9d68', emoji: '◌' },
  happy: { prompt: 'Show us your happy face', color: '#efb83e', emoji: '☺' },
  scared: { prompt: 'Give us your scared face', color: '#9275cc', emoji: '◉' },
  serious: { prompt: 'Show us your serious face', color: '#65718c', emoji: '—' },
  shocked: { prompt: 'Give us your shocked face', color: '#e77862', emoji: '!' },
  stressed: { prompt: 'Show us your stressed face', color: '#b66b82', emoji: '×' },
  tired: { prompt: 'Make your tired face', color: '#6e94d4', emoji: 'z' }
};

const MEME_FILES = {
  angry: ['angry.jpeg', 'angry2.jpg'],
  awkward: ['awkward.jpg'],
  disgusted: ['disgusted 2.jpeg', 'disgusted.jpeg', 'disgusted1.jpg'],
  happy: ['happy.png', 'happy1.jpeg', 'happy2.jpeg'],
  scared: ['scared1.png', 'scareed.jpg'],
  serious: ['serious.jpeg'],
  shocked: ['shocked.webp'],
  stressed: ['stressed2.jpeg'],
  tired: ['tired.jpg']
};
const MEME_BASE = `${import.meta.env.BASE_URL}memeimages/`;
const TRAINING_COUNT = 4;
const MEME_COUNT = 4;

const cosineSimilarity = (left, right) => {
  let dot = 0;
  let leftLength = 0;
  let rightLength = 0;
  for (let index = 0; index < left.length; index += 1) {
    dot += left[index] * right[index];
    leftLength += left[index] ** 2;
    rightLength += right[index] ** 2;
  }
  return dot / (Math.sqrt(leftLength) * Math.sqrt(rightLength) || 1);
};

const selectMeme = (emotion) => {
  const files = MEME_FILES[emotion] || [];
  if (!files.length) return '';
  const file = files[Math.floor(Math.random() * files.length)];
  return `${MEME_BASE}${encodeURIComponent(file).replace(/%2F/g, '/')}`;
};

const downloadStrip = (photos, includeMemes = true) => {
  const canvas = document.createElement('canvas');
  const width = 900;
  const rowHeight = 330;
  canvas.width = width;
  canvas.height = rowHeight * photos.length + 100;
  const context = canvas.getContext('2d');
  context.fillStyle = '#fff8e9';
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = '#302822';
  context.font = '700 28px Space Grotesk, sans-serif';
  context.textAlign = 'center';
  context.fillText('FACE TO MEME LAB', width / 2, 45);
  let loaded = 0;
  const expectedImages = includeMemes ? photos.length * 2 : photos.length;
  photos.forEach((item, index) => {
    const draw = (source, x, y, w, h) => {
      const image = new Image();
      image.onload = () => {
        context.drawImage(image, x, y, w, h);
        loaded += 1;
        if (loaded === expectedImages) {
          const link = document.createElement('a');
          link.download = 'face-to-meme-strip.png';
          link.href = canvas.toDataURL('image/png');
          link.click();
        }
      };
      image.src = source;
    };
    const y = 65 + index * rowHeight;
    draw(item.photo, 24, y, 410, 250);
    if (includeMemes) draw(item.meme, 466, y, 410, 250);
    context.fillStyle = '#806b56';
    context.font = '500 16px DM Mono, monospace';
    context.textAlign = 'left';
    context.fillText(`${index + 1}  ${item.label.toUpperCase()}`, 24, y + 280);
    context.textAlign = 'right';
    context.fillText(includeMemes ? 'participant  ↔  cat' : 'participant training example', 876, y + 280);
  });
};

function App() {
  const [booth, setBooth] = useState('training');
  const [screen, setScreen] = useState('intro');
  const [selectedEmotions, setSelectedEmotions] = useState([]);
  const [trainingPhotos, setTrainingPhotos] = useState([]);
  const [trainingIndex, setTrainingIndex] = useState(0);
  const [memePhotos, setMemePhotos] = useState([]);
  const [cameraState, setCameraState] = useState('idle');
  const [cameraError, setCameraError] = useState('');
  const [countdown, setCountdown] = useState(null);
  const [flash, setFlash] = useState(false);
  const [captureBusy, setCaptureBusy] = useState(false);
  const [notice, setNotice] = useState('');
  const [memeNotice, setMemeNotice] = useState('');
  const [stripReady, setStripReady] = useState(false);
  const [evaluations, setEvaluations] = useState([]);
  const [evaluationIndex, setEvaluationIndex] = useState(0);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const modelRef = useRef(null);
  const countdownRef = useRef(null);

  const currentEmotion = selectedEmotions[trainingIndex];
  const totalTraining = trainingPhotos.length;
  const modelReady = trainingPhotos.length === TRAINING_COUNT;
  const sessionData = useMemo(() => ({
    selectedTrainingEmotions: selectedEmotions,
    trainingPhotos: trainingPhotos.map(({ photo, label }) => ({ photo, label })),
    memeBoothResults: memePhotos.map((item, index) => ({
      participantPhoto: item.photo,
      predictedEmotion: item.label,
      memeImage: item.meme,
      participantAgreed: evaluations[index]?.accurate ?? null
    })),
    agreementCount: evaluations.filter(item => item.accurate).length,
    agreementPercentage: evaluations.length ? Math.round((evaluations.filter(item => item.accurate).length / evaluations.length) * 100) : 0,
    finalReflection: ''
  }), [selectedEmotions, trainingPhotos, memePhotos, evaluations]);

  const attachStream = useCallback(async () => {
    if (!streamRef.current || !videoRef.current) return;
    videoRef.current.srcObject = streamRef.current;
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
      setCameraError('This browser does not support webcam access.');
      return false;
    }
    setCameraState('loading');
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
        : 'We could not start the camera. Check that another app is using it.');
      return false;
    }
  }, [attachStream]);

  useEffect(() => {
    startCamera();
    return () => {
      streamRef.current?.getTracks().forEach(track => track.stop());
      clearInterval(countdownRef.current);
    };
  }, [startCamera]);

  useEffect(() => { attachStream(); }, [booth, attachStream]);

  const getFeatureVector = async () => {
    if (!modelRef.current) modelRef.current = await mobilenet.load({ version: 2, alpha: 1 });
    const tensor = modelRef.current.infer(videoRef.current, true);
    const vector = Array.from(await tensor.data());
    tensor.dispose();
    return vector;
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
    countdownRef.current = setInterval(() => {
      remaining -= 1;
      if (remaining <= 0) {
        clearInterval(countdownRef.current);
        setCountdown(null);
        setFlash(true);
        window.setTimeout(() => setFlash(false), 180);
        onCapture();
      } else setCountdown(remaining);
    }, 1000);
  }, [captureBusy, startCamera]);

  const captureTraining = async () => {
    try {
      const photo = captureDataUrl();
      let vector = null;
      try {
        vector = await Promise.race([
          getFeatureVector(),
          new Promise((_, reject) => window.setTimeout(() => reject(new Error('model-timeout')), 3000))
        ]);
      } catch {
        setNotice('Photo saved. Local model matching is unavailable, so the Meme Booth will use your selected labels.');
      }
      setTrainingPhotos(current => [...current, { photo, label: currentEmotion, vector }]);
      if (trainingIndex + 1 >= TRAINING_COUNT) {
        setNotice('All four personal examples are ready. Enter the Meme Booth when you are ready.');
        setScreen('trainingComplete');
      } else {
        setTrainingIndex(index => index + 1);
      }
    } catch {
      setNotice('This photo could not be captured. Keep your face in frame and try again.');
    } finally {
      setCaptureBusy(false);
    }
  };

  const classifyExpression = async () => {
    const vector = await getFeatureVector();
    const vectorSamples = trainingPhotos.filter(sample => sample.vector);
    if (!vectorSamples.length) {
      return { label: selectedEmotions[memePhotos.length % selectedEmotions.length], score: 0 };
    }
    const closest = vectorSamples.reduce((best, sample) => {
      const score = cosineSimilarity(vector, sample.vector);
      return score > best.score ? { label: sample.label, score } : best;
    }, { label: selectedEmotions[0], score: -Infinity });
    return closest;
  };

  const captureMeme = async () => {
    try {
      const match = await classifyExpression();
      const item = {
        photo: captureDataUrl(),
        label: match.label,
        confidence: match.score,
        meme: selectMeme(match.label)
      };
      setMemePhotos(current => {
        const next = [...current, item];
        if (next.length === MEME_COUNT) setStripReady(true);
        if (next.length === MEME_COUNT) {
          setEvaluations(next.map(item => ({ ...item, accurate: null })));
          setEvaluationIndex(0);
          setScreen('evaluate');
        }
        return next;
      });
      setMemeNotice(`Looks ${match.label}. Matched to a local ${match.label} cat meme.`);
    } catch {
      const fallbackLabel = selectedEmotions[memePhotos.length % selectedEmotions.length];
      const fallbackPhoto = {
        photo: captureDataUrl(),
        label: fallbackLabel,
        confidence: 0,
        meme: selectMeme(fallbackLabel)
      };
      setMemePhotos(current => {
        const next = [...current, fallbackPhoto];
        if (next.length === MEME_COUNT) setStripReady(true);
        if (next.length === MEME_COUNT) {
          setEvaluations(next.map(item => ({ ...item, accurate: null })));
          setEvaluationIndex(0);
          setScreen('evaluate');
        }
        return next;
      });
      setMemeNotice(`Matched this pose to your trained ${fallbackLabel} example.`);
    } finally {
      setCaptureBusy(false);
    }
  };

  const toggleEmotion = (emotion) => {
    setSelectedEmotions(current => current.includes(emotion)
      ? current.filter(item => item !== emotion)
      : current.length < TRAINING_COUNT ? [...current, emotion] : current);
  };

  const enterMemeBooth = () => {
    if (!modelReady) return;
    setBooth('meme');
    setMemePhotos([]);
    setStripReady(false);
    setMemeNotice('');
    setScreen('memeIntro');
  };

  const recordEvaluation = (accurate) => {
    setEvaluations(current => current.map((item, index) => index === evaluationIndex ? { ...item, accurate } : item));
    if (evaluationIndex + 1 >= MEME_COUNT) setScreen('results');
    else setEvaluationIndex(index => index + 1);
  };

  const reset = () => {
    setBooth('training');
    setScreen('intro');
    setSelectedEmotions([]);
    setTrainingPhotos([]);
    setTrainingIndex(0);
    setMemePhotos([]);
    setStripReady(false);
    setEvaluations([]);
    setEvaluationIndex(0);
    setNotice('');
    setMemeNotice('');
  };

  return <main className={`app-shell ${booth === 'meme' ? 'meme-mode' : ''}`}>
    <header className="topbar">
      <div className="brand"><div className="brand-mark"><Sparkles size={18} /></div><span>face to meme lab</span></div>
      <div className="privacy-chip"><LockKeyhole size={14} /> camera + training stay local</div>
    </header>
    <div className="booth-switcher">
      <button className={booth === 'training' ? 'active' : ''} onClick={() => setBooth('training')}><span>01</span> training booth</button>
      <i /><button className={booth === 'meme' ? 'active' : ''} disabled={!modelReady} onClick={enterMemeBooth}><span>02</span> meme booth</button>
    </div>
    {screen === 'intro' && <IntroScreen onStart={() => setScreen('selection')} />}
    {screen === 'selection' && <TrainingSelection {...{ selectedEmotions, toggleEmotion }} onContinue={() => setScreen('trainingIntro')} />}
    {screen === 'trainingIntro' && <InstructionScreen eyebrow="booth one / build the brain" heading="Teach the AI" text="You'll take one photo for each emotion you selected. Use your facial expression to show the emotion as naturally as you can." detail={`Training 1 of 4`} button="Start Training Photos" onClick={() => setScreen('training')} />}
    {screen === 'training' && <TrainingBooth {...{ selectedEmotions, toggleEmotion, currentEmotion, trainingIndex, trainingPhotos, captureTraining, runCountdown, captureBusy, countdown, flash, cameraState, cameraError, startCamera, videoRef, enterMemeBooth, notice, modelReady, reset }} />}
    {screen === 'trainingComplete' && <TransitionScreen heading="Training Complete" text="You've shown the AI how you express four different emotions. Now let's see how well it can read you without being told what emotion you're making." button="Enter the Meme Booth" onClick={enterMemeBooth} photos={trainingPhotos} onDownload={() => downloadStrip(trainingPhotos, false)} />}
    {screen === 'memeIntro' && <InstructionScreen eyebrow="booth two / make the meme" heading="Now Put the AI to the Test" text="Pose however you want. This time, we won't tell you which emotion to make. The AI will interpret your expression and pair your photo with a cat meme that it thinks matches your emotion." detail="Will it get you right?" button="Start Meme Booth" onClick={() => setScreen('meme')} />}
    {screen === 'meme' && <MemeBooth {...{ memePhotos, captureMeme, runCountdown, captureBusy, countdown, flash, cameraState, cameraError, startCamera, videoRef, memeNotice, stripReady, reset }} />}
    {screen === 'evaluate' && <EvaluationScreen item={evaluations[evaluationIndex]} index={evaluationIndex} onEvaluate={recordEvaluation} />}
    {screen === 'results' && <ResultsScreen {...{ memePhotos, evaluations, sessionData, reset }} />}
    <footer><span><Users size={14} /> {trainingPhotos.length} personal training examples</span><span>Predictions are interpretations of expressions, not feelings.</span></footer>
  </main>;
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
      {flash && <div className="flash" />}{children}
    </div>
    <div className="camera-base"><span className="live-dot" /> live mirror <span><LockKeyhole size={12} /> never uploaded</span></div>
  </div>;
}

function IntroScreen({ onStart }) {
  return <section className="instruction-card"><div className="eyebrow"><span className="eyebrow-dot" /> a tiny expression experiment</div><h1>Can AI Read <em>Your Emotions?</em></h1><p>AI can recognize patterns in our expressions, but does it actually understand how we feel?</p><p>First, you'll help train the system by showing it how you express different emotions. Then, we'll put it to the test.</p><button className="button yellow" onClick={onStart}>Start Training <ChevronRight size={17} /></button></section>;
}

function TrainingSelection({ selectedEmotions, toggleEmotion, onContinue }) {
  return <section className="booth-page"><div className="booth-intro"><div className="eyebrow"><span className="eyebrow-dot" /> step one / choose your labels</div><h1>Choose your <em>emotions.</em></h1><p>Choose 4 emotions to teach the AI.</p></div><EmotionPicker {...{ selectedEmotions, toggleEmotion }} /><div className="selection-footer"><b>{selectedEmotions.length}/4 selected</b><span>Everyone expresses emotions differently. Choose four emotions and show the AI what they look like on you.</span>{selectedEmotions.length === TRAINING_COUNT && <button className="button yellow" onClick={onContinue}>Teach the AI <ChevronRight size={17} /></button>}</div></section>;
}

function InstructionScreen({ eyebrow, heading, text, detail, button, onClick }) {
  return <section className="instruction-card"><div className="eyebrow"><span className="eyebrow-dot purple" /> {eyebrow}</div><h1>{heading}</h1><p>{text}</p><strong className="instruction-detail">{detail}</strong><button className="button yellow" onClick={onClick}>{button} <ChevronRight size={17} /></button></section>;
}

function TransitionScreen({ heading, text, button, onClick, photos, onDownload }) {
  return <section className="instruction-card"><div className="eyebrow"><span className="eyebrow-dot" /> your training set is ready</div><h1>{heading}</h1><p>{text}</p><div className="transition-actions"><button className="button dark" onClick={onClick}>{button} <ChevronRight size={17} /></button><button className="button cream dark-text" onClick={onDownload}><Download size={17} /> Download Training Strip</button></div><PhotoStrip photos={photos} type="training" /></section>;
}

function EvaluationScreen({ item, index, onEvaluate }) {
  return <section className="evaluation-card"><div className="eyebrow"><span className="eyebrow-dot purple" /> evaluate the interpretation</div><h1>AI thinks you're: <em>{item?.label}</em></h1><p className="evaluation-progress">{index + 1} of 4 evaluated</p><div className="evaluation-pair"><img src={item?.photo} alt={`Participant photo ${index + 1}`} /><img src={item?.meme} alt={`${item?.label} cat meme`} /></div><h2>Did the AI get this one right?</h2><div className="evaluation-actions"><button className="button yes" onClick={() => onEvaluate(true)}>👍 Yes</button><button className="button no" onClick={() => onEvaluate(false)}>👎 No</button></div></section>;
}

function ResultsScreen({ memePhotos, evaluations, reset }) {
  const agreementCount = evaluations.filter(item => item.accurate).length;
  return <section className="results-screen"><div className="booth-intro"><div className="eyebrow"><span className="eyebrow-dot" /> results / your feedback loop</div><h1>The AI got <em>{agreementCount} out of 4</em> right.</h1><p>You agreed with the AI {agreementCount * 25}% of the time.</p></div><PhotoStrip photos={memePhotos} type="meme" /><p className="closing-message"><b>Thanks for training the AI.</b><br />Your expressions helped teach the system, and your feedback helped evaluate its interpretations.</p><div className="strip-actions"><button className="button dark" onClick={() => downloadStrip(memePhotos)}><Download size={17} /> Download Photo Strip</button><button className="button cream dark-text" onClick={reset}><RefreshCw size={16} /> Start another participant</button></div></section>;
}

function TrainingBooth(props) {
  const { selectedEmotions, toggleEmotion, currentEmotion, trainingIndex, trainingPhotos, runCountdown, captureBusy, countdown, flash, cameraState, cameraError, startCamera, videoRef, enterMemeBooth, notice, modelReady, reset } = props;
  const choosing = selectedEmotions.length < TRAINING_COUNT;
  const complete = trainingPhotos.length === TRAINING_COUNT;
  return <section className="booth-page">
    <div className="booth-intro"><div className="eyebrow"><span className="eyebrow-dot" /> booth one / build the brain</div><h1>Teach it <em>your face.</em></h1><p>Choose four emotions, then show the booth how you personally express each one.</p></div>
    {choosing ? <EmotionPicker {...{ selectedEmotions, toggleEmotion }} /> : <div className="booth-layout">
      <div><BoothCamera {...props} theme="training-curtain"><div className="booth-prompt"><span>training photo {trainingIndex + 1} of 4</span><strong>{complete ? 'Training complete!' : EMOTION_META[currentEmotion].prompt}</strong><small>{complete ? 'Your personal examples are ready.' : 'One photo, one label, no retakes.'}</small></div></BoothCamera>
        <div className="booth-controls"><div className="progress-dots">{selectedEmotions.map((label, index) => <i key={label} className={index <= trainingIndex ? 'on' : ''} style={{ '--dot': EMOTION_META[label].color }} />)}</div>{!complete ? <button className="button yellow wide" onClick={() => runCountdown(() => props.captureTraining?.())} disabled={captureBusy || countdown}><Camera size={18} /> {captureBusy ? 'Saving example…' : `Capture ${currentEmotion}`}</button> : <button className="button yellow wide" onClick={enterMemeBooth}><ChevronRight size={17} /> Enter Meme Booth</button>}</div>
      </div>
      <div className="strip-column">{complete ? <PhotoStrip photos={trainingPhotos} type="training" /> : <EmptyStrip type="training" photos={trainingPhotos} />}<h2 className="thank-you">{complete ? 'Your expressions are trained.' : `${trainingPhotos.length} of 4 examples captured`}</h2>{complete && <div className="strip-actions"><button className="button dark" onClick={enterMemeBooth}>Start posing freely <ChevronRight size={17} /></button><button className="button cream dark-text" onClick={reset}><RefreshCw size={16} /> Start over</button></div>}</div>
    </div>}
    {choosing && <div className="selection-footer"><b>{selectedEmotions.length} of 4 emotions selected</b><span>Pick four different labels to unlock the training camera.</span></div>}
    {notice && <p className="notice">{notice}</p>}
    {modelReady && <div className="model-panel"><div><b>Personal training complete</b><p>Each selected emotion has one labelled example. The Meme Booth will compare new expressions with these examples.</p></div><button className="button dark" onClick={enterMemeBooth}>Continue <ChevronRight size={16} /></button></div>}
  </section>;
}

function EmotionPicker({ selectedEmotions, toggleEmotion }) {
  return <div className="emotion-picker"><div className="picker-heading"><div><b>Choose your four training emotions</b><p>These are the faces the booth will learn from you.</p></div><span>{selectedEmotions.length}/4</span></div><div className="emotion-grid">{EMOTIONS.map(emotion => {
    const selected = selectedEmotions.includes(emotion);
    return <button key={emotion} className={`emotion-option ${selected ? 'selected' : ''}`} onClick={() => toggleEmotion(emotion)} disabled={!selected && selectedEmotions.length >= TRAINING_COUNT} style={{ '--emotion-color': EMOTION_META[emotion].color }}><span className="emotion-emoji">{EMOTION_META[emotion].emoji}</span><span>{emotion}</span>{selected && <Check size={16} />}</button>;
  })}</div></div>;
}

function MemeBooth(props) {
  const { memePhotos, runCountdown, captureBusy, countdown, flash, cameraState, cameraError, startCamera, videoRef, memeNotice, stripReady, reset } = props;
  return <section className="booth-page">
    <div className="booth-intro"><div className="eyebrow"><span className="eyebrow-dot purple" /> booth two / make the meme</div><h1>Pose <em>your way.</em></h1><p>No prompts. The booth interprets your expression and finds its cat counterpart.</p></div>
    <div className="booth-layout">
      <div><BoothCamera {...props} theme="meme-curtain"><div className="booth-prompt"><span>meme photo {memePhotos.length + 1} of 4</span><strong>Make it your own.</strong><small>Pose freely. The model will pick the closest trained emotion.</small></div></BoothCamera><div className="booth-controls"><div className="progress-dots">{[0, 1, 2, 3].map(index => <i key={index} className={index < memePhotos.length ? 'on' : ''} style={{ '--dot': '#8e72c5' }} />)}</div>{!stripReady && <button className="button coral wide" onClick={() => runCountdown(() => props.captureMeme?.())} disabled={captureBusy || countdown}><Camera size={18} /> {captureBusy ? 'Matching your meme…' : 'Take this photo'}</button>}</div></div>
      <div className="strip-column">{stripReady ? <><PhotoStrip photos={memePhotos} type="meme" /><h2 className="thank-you">Your meme strip is ready!</h2><div className="strip-actions"><button className="button dark" onClick={() => downloadStrip(memePhotos)}><Download size={17} /> Download Photo Strip</button><button className="button cream dark-text" onClick={reset}><RefreshCw size={16} /> Train another participant</button></div></> : <EmptyStrip type="meme" photos={memePhotos} />}</div>
    </div>
    {memeNotice && <p className="notice">{memeNotice}</p>}
  </section>;
}

function EmptyStrip({ type, photos = [] }) {
  return <div className={`empty-strip ${type}`}><div className="strip-top"><Sparkles size={12} /> {type === 'meme' ? 'meme strip preview' : 'training examples'}</div>{[0, 1, 2, 3].map(index => <div className="empty-row" key={index}><span>{index + 1}</span><i>{photos[index] ? 'matched' : type === 'meme' ? 'pose + cat' : 'waiting for a face'}</i></div>)}<div className="strip-bottom">face to meme lab</div></div>;
}

function PhotoStrip({ photos, type }) {
  if (!photos.length) return <EmptyStrip type={type} />;
  return <div className={`photo-strip ${type}`}><div className="strip-top"><Sparkles size={12} /> {type === 'meme' ? 'face to meme lab' : 'personal training set'}</div>{photos.map((item, index) => <div className="strip-row" key={`${item.label}-${index}`}><div className="strip-photo"><img src={item.photo} alt={`Participant ${index + 1}`} /><small>{index + 1} · {item.label}</small></div>{type === 'meme' && <div className="strip-meme"><img src={item.meme} alt={`${item.label} cat meme`} /><small>{item.label} match</small></div>}</div>)}<div className="strip-bottom">local, playful, yours</div></div>;
}

createRoot(document.getElementById('root')).render(<App />);

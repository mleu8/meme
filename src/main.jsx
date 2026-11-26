import React, { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import ml5 from 'ml5';
import {
  Camera, Check, ChevronRight, Download, LockKeyhole, RefreshCw, Sparkles,
  Trash2, TriangleAlert, Users
} from 'lucide-react';
import './styles.css';

const EMOTIONS = ['angry', 'awkward', 'disgusted', 'happy', 'scared', 'serious', 'shocked', 'stressed', 'tired'];
const EMOTION_META = {
  angry: { prompt: 'Give us your angry face', color: '#c8754c', emoji: '↯' },
  awkward: { prompt: 'Show us your awkward face', color: '#b68b65', emoji: '⌁' },
  disgusted: { prompt: 'Make your most disgusted face', color: '#887c5a', emoji: '◌' },
  happy: { prompt: 'Show us your happy face', color: '#e6b353', emoji: '☺' },
  scared: { prompt: 'Give us your scared face', color: '#8b7059', emoji: '◉' },
  serious: { prompt: 'Show us your serious face', color: '#5e5147', emoji: '—' },
  shocked: { prompt: 'Give us your shocked face', color: '#d78355', emoji: '!' },
  stressed: { prompt: 'Show us your stressed face', color: '#a75f4e', emoji: '×' },
  tired: { prompt: 'Make your tired face', color: '#9b876b', emoji: 'z' }
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
const PREDICTION_WINDOW_MS = 700;
const PREDICTION_HOLD_MS = 300;
const PREDICTION_SAMPLE_INTERVAL_MS = 100;
const JOURNEY_STAGES = ['Consent', 'Train the model', 'Try the meme booth', 'Download'];

const euclideanDistance = (left, right) => Math.sqrt(
  left.reduce((sum, value, index) => sum + (value - right[index]) ** 2, 0)
);

const point = (face, index) => face?.keypoints?.[index];

const distance = (first, second) => {
  if (!first || !second) return null;
  return Math.hypot(first.x - second.x, first.y - second.y);
};

const validVector = vector => Array.isArray(vector)
  && vector.length === 8
  && vector.every(value => Number.isFinite(value));

const predictLabel = (features, classifier) => {
  if (!validVector(features) || !classifier?.centroids?.length) return null;
  return classifier.centroids.reduce((best, candidate) => {
    const distance = euclideanDistance(features, candidate.average);
    return !best || distance < best.distance
      ? { label: candidate.label, distance }
      : best;
  }, null)?.label ?? null;
};

const waitForVideoFrame = video => {
  if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      window.clearTimeout(timeout);
      video.removeEventListener('loadeddata', check);
      video.removeEventListener('playing', check);
      video.removeEventListener('resize', check);
    };
    const check = () => {
      if (video.readyState < 2 || !video.videoWidth || !video.videoHeight) return;
      cleanup();
      resolve();
    };
    const timeout = window.setTimeout(() => {
      cleanup();
      reject(new Error('Webcam did not produce a video frame'));
    }, 10000);
    video.addEventListener('loadeddata', check);
    video.addEventListener('playing', check);
    video.addEventListener('resize', check);
    check();
  });
};

// Uses MediaPipe FaceMesh indices for the 468-point face mesh.
const extractFaceFeatures = face => {
  if (!face?.keypoints) return null;
  const faceWidth = distance(point(face, 234), point(face, 454));
  if (!faceWidth || faceWidth <= 1) return null;

  const vector = [
    distance(point(face, 61), point(face, 291)) / faceWidth,
    distance(point(face, 13), point(face, 14)) / faceWidth,
    distance(point(face, 159), point(face, 145)) / faceWidth,
    distance(point(face, 386), point(face, 374)) / faceWidth,
    distance(point(face, 105), point(face, 159)) / faceWidth,
    distance(point(face, 334), point(face, 386)) / faceWidth,
    distance(point(face, 61), point(face, 13)) / faceWidth,
    distance(point(face, 291), point(face, 14)) / faceWidth
  ];
  return validVector(vector) ? vector : null;
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
  context.fillStyle = '#fffaf1';
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = '#49372d';
  context.font = '800 30px Baloo 2, sans-serif';
  context.textAlign = 'center';
  context.fillText('PURRCEPTION', width / 2, 45);
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
          link.download = 'purrception-photo-strip.png';
          link.href = canvas.toDataURL('image/png');
          link.click();
        }
      };
      image.src = source;
    };
    const y = 65 + index * rowHeight;
    draw(item.photo, 24, y, 410, 250);
    if (includeMemes) draw(item.meme, 466, y, 410, 250);
    context.fillStyle = '#786d5d';
    context.font = '500 16px DM Mono, monospace';
    context.textAlign = 'left';
    context.fillText(`${index + 1}  ${item.label.toUpperCase()}`, 24, y + 280);
    context.textAlign = 'right';
    context.fillText(includeMemes ? 'participant  ↔  cat' : 'participant training example', 876, y + 280);
  });
};

function CatDoodle({ className = '' }) {
  const catId = useId().replace(/:/g, '');
  const catPath = 'M61 249c-13 0-21-9-24-22-5-20-4-47 5-67-11-10-17-26-17-42 0-23 13-40 32-49l13-42c2-13 12-14 19-2l22 30c13-5 30-6 43 0l25-29c7-10 17-7 18 5l6 42c12 13 18 30 18 50 0 21-7 39-16 54 12 17 20 38 21 54 0 12-8 19-21 19h-66c-12 0-19-8-15-20 5-17 18-36 28-55-12 19-24 45-35 63-5 9-13 11-23 11Z';
  const tailPath = 'M198 228c18 4 31-1 35-13 4-13-4-25 2-38 5-13 17-17 26-10 9 7 9 22 5 36-7 23-22 39-41 45-12 4-25 1-32-7Z';
  return <svg className={`cat-doodle ${className}`} viewBox="0 0 270 265" aria-hidden="true" focusable="false">
    <defs>
      <clipPath id={`${catId}-cat`}><path d={catPath} /></clipPath>
      <clipPath id={`${catId}-tail`}><path d={tailPath} /></clipPath>
    </defs>
    <ellipse className="cat-shadow" cx="128" cy="252" rx="111" ry="8" />
    <path className="cat-tail" d={tailPath} />
    <path className="cat-tail-band" clipPath={`url(#${catId}-tail)`} d="M220 228c14-5 20-15 17-26-2-8-4-15-1-23m18 50c9-10 15-22 17-36" />
    <path className="cat-silhouette" d={catPath} />
    <path className="cat-patch-orange" clipPath={`url(#${catId}-cat)`} d="M111 56c15-6 29-4 43 1l24-28c7-9 14-8 15 4l6 43c13 14 18 31 18 51 0 20-7 38-17 53 12 17 19 37 20 54 0 10-7 15-18 15-13 0-21-7-27-19-7-14-4-27 4-41 10-17 14-35 5-50-9-15-25-20-45-24-19-4-30-15-31-31-2-12 0-21 3-28Z" />
    <path className="cat-patch-orange" clipPath={`url(#${catId}-cat)`} d="M82 64c12-6 24-8 38-5-8 12-20 18-38 17-5-3-5-8 0-12Z" />
    <path className="cat-patch-cream" clipPath={`url(#${catId}-cat)`} d="M142 218c15 0 26 9 34 21-13 9-29 11-45 5 8-7 12-15 11-26Z" />
    <path className="cat-ear-inner" d="m75 41 7-22 19 27Z" />
    <path className="cat-ear-inner" d="m169 47 18-22 5 34Z" />
    <path className="cat-leg-line" d="M103 175c-8 23-12 49-18 70m72-69c-13 22-25 47-37 70" />
    <circle className="cat-eye" cx="72" cy="99" r="8" />
    <circle className="cat-eye" cx="153" cy="99" r="8" />
    <path className="cat-nose" d="m108 111 6-4 6 4-6 6Z" />
    <path className="cat-mouth" d="M114 117v8m0 0c-8 10-17 8-22 0m22 0c8 10 17 8 22 0" />
    <path className="cat-whiskers" d="m53 111-24-5m24 14-25 1m126-8 24 2m-24 9 24 7" />
    <path className="cat-outline" d={catPath} />
  </svg>;
}

function App() {
  const [booth, setBooth] = useState('training');
  const [screen, setScreen] = useState('intro');
  const [selectedEmotions, setSelectedEmotions] = useState([]);
  const [trainingPhotos, setTrainingPhotos] = useState([]);
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
  const [trainingData, setTrainingData] = useState([]);
  const [classifierReady, setClassifierReady] = useState(false);
  const [hasConsent, setHasConsent] = useState(false);
  const [faceMeshReady, setFaceMeshReady] = useState(false);
  const [debugMode, setDebugMode] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);
  const [collectingTraining, setCollectingTraining] = useState(false);
  const [trainingCaptureStatus, setTrainingCaptureStatus] = useState('ready');
  const [currentFeatures, setCurrentFeatures] = useState(null);
  const [rawPrediction, setRawPrediction] = useState(null);
  const [currentPredictedEmotion, setCurrentPredictedEmotion] = useState(null);
  const [currentDisplayedMeme, setCurrentDisplayedMeme] = useState('');
  const [loadedMeme, setLoadedMeme] = useState('');
  const [predictionHistory, setPredictionHistory] = useState([]);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const faceMeshRef = useRef(null);
  const trackedVideoRef = useRef(null);
  const countdownRef = useRef(null);
  const collectingRef = useRef(false);
  const collectingModeRef = useRef(null);
  const faceMeshLoadingRef = useRef(false);
  const faceMeshInitializationRef = useRef(0);
  const trainingSampleBufferRef = useRef([]);
  const lastSampleTimeRef = useRef(0);
  const currentFeaturesRef = useRef(null);
  const faceDetectedRef = useRef(false);
  const classifierRef = useRef(null);
  const screenRef = useRef(screen);
  const rawPredictionRef = useRef(null);
  const currentPredictedEmotionRef = useRef(null);
  const currentDisplayedMemeRef = useRef('');
  const predictionHistoryRef = useRef([]);
  const predictionCandidateRef = useRef(null);
  const captureInProgressRef = useRef(false);
  screenRef.current = screen;

  const trainingIndex = trainingPhotos.length;
  const currentEmotion = selectedEmotions[trainingIndex];
  const totalTraining = trainingPhotos.length;
  const modelReady = classifierReady && trainingPhotos.length === TRAINING_COUNT;
  const clearLivePrediction = useCallback(() => {
    if (!rawPredictionRef.current
      && !currentPredictedEmotionRef.current
      && !currentDisplayedMemeRef.current
      && !predictionHistoryRef.current.length) return;
    rawPredictionRef.current = null;
    currentPredictedEmotionRef.current = null;
    currentDisplayedMemeRef.current = '';
    predictionHistoryRef.current = [];
    predictionCandidateRef.current = null;
    setRawPrediction(null);
    setCurrentPredictedEmotion(null);
    setCurrentDisplayedMeme('');
    setLoadedMeme('');
    setPredictionHistory([]);
  }, []);
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
    if (videoRef.current.srcObject !== streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
    }
    if (videoRef.current.paused) {
      await videoRef.current.play().catch(() => {});
    }
  }, []);

  const startCamera = useCallback(async () => {
    if (!hasConsent) return false;
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
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 720, height: 540, facingMode: 'user' },
        audio: false
      });
      if (!hasConsent || (screenRef.current !== 'training' && screenRef.current !== 'meme')) {
        stream.getTracks().forEach(track => track.stop());
        setCameraState('idle');
        return false;
      }
      streamRef.current = stream;
      await attachStream();
      setCameraState('ready');
      return true;
    } catch (error) {
      if (screenRef.current !== 'training' && screenRef.current !== 'meme') {
        setCameraState('idle');
        return false;
      }
      setCameraState('error');
      setCameraError(error.name === 'NotAllowedError'
        ? 'Camera access was blocked. Allow it in your browser settings, then try again.'
        : 'We could not start the camera. Check that another app is using it.');
      return false;
    }
  }, [attachStream, hasConsent]);

  const stopFaceTracking = useCallback(() => {
    if (faceMeshRef.current?.detectStop) faceMeshRef.current.detectStop();
  }, []);

  const stopCamera = useCallback(() => {
    faceMeshInitializationRef.current += 1;
    faceMeshLoadingRef.current = false;
    collectingRef.current = false;
    collectingModeRef.current = null;
    trainingSampleBufferRef.current = [];
    clearInterval(countdownRef.current);
    countdownRef.current = null;
    stopFaceTracking();
    faceMeshRef.current?.dispose?.();
    faceMeshRef.current = null;
    trackedVideoRef.current = null;
    streamRef.current?.getTracks().forEach(track => track.stop());
    streamRef.current = null;
    faceDetectedRef.current = false;
    currentFeaturesRef.current = null;
    setCameraState('idle');
    setCameraError('');
    setCountdown(null);
    setCaptureBusy(false);
    setCollectingTraining(false);
    setFaceMeshReady(false);
    setFaceDetected(false);
    setCurrentFeatures(null);
    clearLivePrediction();
  }, [clearLivePrediction, stopFaceTracking]);

  useEffect(() => {
    return () => {
      faceMeshInitializationRef.current += 1;
      stopFaceTracking();
      faceMeshRef.current?.dispose?.();
      streamRef.current?.getTracks().forEach(track => track.stop());
      clearInterval(countdownRef.current);
    };
  }, [stopFaceTracking]);

  useEffect(() => {
    const cameraScreen = hasConsent && (screen === 'training' || screen === 'meme');
    if (cameraScreen) {
      if (cameraState === 'idle') startCamera();
      return;
    }
    if (streamRef.current || faceMeshRef.current) stopCamera();
  }, [cameraState, hasConsent, screen, startCamera, stopCamera]);

  useEffect(() => { attachStream(); }, [booth, screen, attachStream]);

  const getCurrentFaceFeatures = useCallback(() => currentFeaturesRef.current, []);

  const startFaceTracking = useCallback((video = videoRef.current) => {
    if (!faceMeshRef.current || !video) return;
    if (typeof faceMeshRef.current.detectStart !== 'function') return;
    if (trackedVideoRef.current === video && faceMeshRef.current.detecting) return;
    faceMeshRef.current.detectStart(video, results => {
      const face = results?.[0];
      const features = face ? extractFaceFeatures(face) : null;
      const hasValidFace = Boolean(features);
      faceDetectedRef.current = hasValidFace;
      setFaceDetected(hasValidFace);
      currentFeaturesRef.current = features;
      if (!features) {
        setCurrentFeatures(null);
        if (screenRef.current === 'meme') clearLivePrediction();
        return;
      }

      const now = performance.now();
      if (now - lastSampleTimeRef.current < PREDICTION_SAMPLE_INTERVAL_MS) return;
      lastSampleTimeRef.current = now;
      setCurrentFeatures(features);

      if (screenRef.current === 'meme' && classifierRef.current) {
        const label = predictLabel(features, classifierRef.current);
        rawPredictionRef.current = label;
        setRawPrediction(label);
        if (label) {
          const history = predictionHistoryRef.current
            .filter(sample => now - sample.timestamp <= PREDICTION_WINDOW_MS);
          history.push({ label, timestamp: now });
          predictionHistoryRef.current = history;
          setPredictionHistory(history.map(sample => sample.label));

          const counts = history.reduce((totals, sample) => {
            totals[sample.label] = (totals[sample.label] || 0) + 1;
            return totals;
          }, {});
          const dominant = Object.entries(counts)
            .sort((left, right) => right[1] - left[1])[0];
          const isDominant = history.length >= 3 && dominant[1] / history.length >= 0.6;
          if (!isDominant) {
            predictionCandidateRef.current = null;
          } else if (predictionCandidateRef.current?.label !== dominant[0]) {
            predictionCandidateRef.current = { label: dominant[0], since: now };
          } else if (now - predictionCandidateRef.current.since >= PREDICTION_HOLD_MS
            && currentPredictedEmotionRef.current !== dominant[0]) {
            const meme = selectMeme(dominant[0]);
            currentPredictedEmotionRef.current = dominant[0];
            currentDisplayedMemeRef.current = meme;
            setCurrentPredictedEmotion(dominant[0]);
            setCurrentDisplayedMeme(meme);
            setLoadedMeme('');
            setMemeNotice('');
          }
        }
      }

      if (!collectingRef.current) return;
      if (collectingModeRef.current === 'training') {
        trainingSampleBufferRef.current.push(features);
      }
    });
    trackedVideoRef.current = video;
  }, [clearLivePrediction]);

  const initializeFaceMesh = useCallback(async () => {
    if (faceMeshRef.current || faceMeshLoadingRef.current || !videoRef.current) return;
    faceMeshLoadingRef.current = true;
    const video = videoRef.current;
    const initializationId = ++faceMeshInitializationRef.current;
    try {
      await attachStream();
      await waitForVideoFrame(video);
      try {
        const webglReady = await ml5.tf.setBackend('webgl');
        if (!webglReady) throw new Error('WebGL backend unavailable');
      } catch (error) {
        console.warn('WebGL unavailable for FaceMesh; falling back to CPU.', error);
        const cpuReady = await ml5.tf.setBackend('cpu');
        if (!cpuReady) throw new Error('No TensorFlow.js backend is available');
      }
      const options = {
        runtime: 'mediapipe',
        maxFaces: 1,
        refineLandmarks: true,
        flipHorizontal: true
      };
      const model = ml5.faceMesh(options);
      await model.ready;
      if (initializationId !== faceMeshInitializationRef.current
        || videoRef.current !== video
        || (screenRef.current !== 'training' && screenRef.current !== 'meme')) {
        model.dispose?.();
        return;
      }
      faceMeshRef.current = model;
      startFaceTracking(video);
      setFaceMeshReady(true);
    } catch (error) {
      if (initializationId !== faceMeshInitializationRef.current) return;
      console.error('FaceMesh initialization failed:', error);
      setNotice('Face tracking could not start. Refresh the page and try again.');
    } finally {
      if (initializationId === faceMeshInitializationRef.current) {
        faceMeshLoadingRef.current = false;
      }
    }
  }, [attachStream, startFaceTracking]);

  useEffect(() => {
    if (!hasConsent || cameraState !== 'ready' || (screen !== 'training' && screen !== 'meme')) return;
    const video = videoRef.current;
    if (!video) return;
    if (!faceMeshRef.current) {
      initializeFaceMesh();
      return;
    }
    if (trackedVideoRef.current === video && faceMeshRef.current.detecting) return;

    let cancelled = false;
    setFaceMeshReady(false);
    faceDetectedRef.current = false;
    setFaceDetected(false);
    currentFeaturesRef.current = null;
    setCurrentFeatures(null);
    clearLivePrediction();
    const reconnectTracking = async () => {
      try {
        await attachStream();
        await waitForVideoFrame(video);
        if (cancelled || videoRef.current !== video) return;
        startFaceTracking(video);
        setFaceMeshReady(true);
      } catch (error) {
        console.error('FaceMesh video reconnection failed:', error);
        if (screenRef.current === 'meme') {
          setMemeNotice('Face tracking could not reconnect. Please return to the Meme Booth and try again.');
        } else {
          setNotice('Face tracking could not reconnect. Please try again.');
        }
      }
    };
    reconnectTracking();
    return () => { cancelled = true; };
  }, [attachStream, cameraState, clearLivePrediction, hasConsent, initializeFaceMesh, screen, startFaceTracking]);

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
    if (!hasConsent) return;
    if (captureBusy) return;
    if (!faceMeshReady) {
      setNotice('Facial tracking is still starting. Try again in a moment.');
      return;
    }
    if (!(await startCamera()) || !videoRef.current?.videoWidth) {
      setNotice('The camera is not ready yet. Allow camera access and try again.');
      return;
    }
    setCaptureBusy(true);
    setNotice('');
    collectingRef.current = false;
    collectingModeRef.current = 'training';
    trainingSampleBufferRef.current = [];
    setCollectingTraining(false);
    setTrainingCaptureStatus('ready');
    setCountdown(3);
    let remaining = 3;
    countdownRef.current = setInterval(() => {
      remaining -= 1;
      // Start collecting facial features during the final 2 seconds.
      if (remaining === 2) {
        collectingRef.current = true;
        setCollectingTraining(booth === 'training');
        setTrainingCaptureStatus('capturing');
      }
      if (remaining <= 0) {
        clearInterval(countdownRef.current);
        collectingRef.current = false;
        setCollectingTraining(false);
        setCountdown(null);
        setFlash(true);
        window.setTimeout(() => setFlash(false), 180);
        onCapture();
      } else setCountdown(remaining);
    }, 1000);
  }, [booth, captureBusy, faceMeshReady, hasConsent, startCamera]);

  const captureTraining = async () => {
    if (!hasConsent) return;
    try {
      const photo = captureDataUrl();
      const samples = [...trainingSampleBufferRef.current];
      if (!samples.length) {
        setNotice('No face detected. Let’s try that one again.');
        setTrainingCaptureStatus('ready');
        setCaptureBusy(false);
        return;
      }
      const rows = samples.map(features => ({ label: currentEmotion, features }));
      setTrainingData(current => [...current, ...rows]);
      setTrainingPhotos(current => [...current, { photo, label: currentEmotion }]);
      setTrainingCaptureStatus('complete');
      trainingSampleBufferRef.current = [];
      if (trainingPhotos.length + 1 >= TRAINING_COUNT) {
        setNotice('All four training photos are ready. Your examples are being added to the shared local classifier.');
        setScreen('trainingComplete');
      }
    } catch {
      setNotice('This photo could not be captured. Keep your face in frame and try again.');
      setTrainingCaptureStatus('ready');
    } finally {
      setCaptureBusy(false);
    }
  };

  useEffect(() => {
    if (!selectedEmotions.length) return;
    const complete = selectedEmotions.every(label =>
      trainingData.some(sample => sample.label === label)
    );
    if (!complete) return;

    const centroids = selectedEmotions.map(label => {
      const samples = trainingData.filter(sample => sample.label === label);
      const average = samples[0].features.map((_, featureIndex) =>
        samples.reduce((sum, sample) => sum + sample.features[featureIndex], 0) / samples.length
      );
      return { label, average };
    });
    classifierRef.current = { centroids };
    setClassifierReady(true);
  }, [trainingData, selectedEmotions]);

  const captureMeme = () => {
    if (!hasConsent) return;
    if (captureInProgressRef.current) return;
    const label = currentPredictedEmotionRef.current;
    const meme = currentDisplayedMemeRef.current;
    if (!faceDetectedRef.current || !label || !meme) {
      setMemeNotice('Move into frame and wait for a stable expression reading before taking a photo.');
      return;
    }
    if (loadedMeme !== meme) {
      setMemeNotice('Your matching meme is still loading. Try again in a moment.');
      return;
    }

    captureInProgressRef.current = true;
    setCaptureBusy(true);
    try {
      const item = {
        photo: captureDataUrl(),
        label,
        meme
      };
      const nextPhotos = [...memePhotos, item];
      setMemePhotos(nextPhotos);
      setMemeNotice(`Saved: AI read your expression as ${label}.`);
      setFlash(true);
      window.setTimeout(() => setFlash(false), 180);
      clearLivePrediction();
      if (nextPhotos.length >= MEME_COUNT) {
        setStripReady(true);
        setEvaluations(nextPhotos.map(photo => ({ ...photo, accurate: null })));
        setEvaluationIndex(0);
        setScreen('evaluate');
      }
    } catch (error) {
      console.error('Meme photo capture failed:', error);
      setMemeNotice('This photo could not be captured. Keep your face in frame and try again.');
    } finally {
      window.setTimeout(() => {
        captureInProgressRef.current = false;
        setCaptureBusy(false);
      }, 250);
    }
  };

  const handleLiveMemeLoad = useCallback(source => {
    if (currentDisplayedMemeRef.current === source) setLoadedMeme(source);
  }, []);

  const handleLiveMemeError = useCallback(() => {
    setLoadedMeme('');
    setMemeNotice('The matching cat meme could not be loaded. Please try again.');
  }, []);

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
    clearLivePrediction();
    setScreen('memeIntro');
  };

  const recordEvaluation = (accurate) => {
    setEvaluations(current => current.map((item, index) => index === evaluationIndex ? { ...item, accurate } : item));
    if (evaluationIndex + 1 >= MEME_COUNT) setScreen('results');
    else setEvaluationIndex(index => index + 1);
  };

  const reset = () => {
    stopCamera();
    setHasConsent(false);
    setBooth('training');
    setScreen('intro');
    setSelectedEmotions([]);
    setTrainingPhotos([]);
    setTrainingCaptureStatus('ready');
    setClassifierReady(false);
    classifierRef.current = null;
    setMemePhotos([]);
    setStripReady(false);
    setMemeNotice('');
    captureInProgressRef.current = false;
    setEvaluations([]);
    setEvaluationIndex(0);
    setNotice('');
    setMemeNotice('');
  };

  const journeyStage = ['intro', 'consent', 'declined'].includes(screen) ? 0
    : ['selection', 'trainingIntro', 'training', 'trainingComplete'].includes(screen) ? 1
      : ['memeIntro', 'meme'].includes(screen) ? 2 : 3;

  return <main className={`app-shell ${booth === 'meme' ? 'meme-mode' : ''}`}>
    <header className="topbar">
      <div className="brand"><div className="brand-mark"><CatDoodle className="brand-cat" /></div><span>Purrception</span></div>
      <div className="privacy-chip"><LockKeyhole size={14} /> camera analysis stays in this browser</div>
    </header>
    <JourneyProgress currentStage={journeyStage} />
    <button className="debug-toggle" onClick={() => setDebugMode(value => !value)}>
      Debug: {debugMode ? 'on' : 'off'}
    </button>
    {debugMode && <aside className="debug-panel">
      <b>Face tracking debug</b>
      <span>Face detected: {faceDetected ? 'yes' : 'no'}</span>
      <span>Current training emotion: {currentEmotion || 'none'}</span>
      <span>Countdown: {countdown ?? 'none'}</span>
      <span>Collecting training data: {collectingTraining ? 'yes' : 'no'}</span>
      <span>Samples for current emotion: {trainingData.filter(item => item.label === currentEmotion).length}</span>
      <span>Total samples per emotion: {selectedEmotions.map(label =>
        `${label}=${trainingData.filter(item => item.label === label).length}`
      ).join(', ') || 'none'}</span>
      <span>Feature vector: {currentFeatures ? currentFeatures.map(value => value.toFixed(3)).join(', ') : 'none'}</span>
      <span>Classifier trained: {classifierReady ? 'yes' : 'no'}</span>
      <span>FaceMesh ready: {faceMeshReady ? 'yes' : 'no'}</span>
      <span>Raw prediction: {rawPrediction || 'none'}</span>
      <span>Stable prediction: {currentPredictedEmotion || 'none'}</span>
      <span>Current displayed meme: {currentDisplayedMeme ? decodeURIComponent(currentDisplayedMeme.split('/').pop()) : 'none'}</span>
      <span>Prediction history: [{predictionHistory.join(', ')}]</span>
      <span>Capture count: {memePhotos.length} / {MEME_COUNT}</span>
    </aside>}
    {screen === 'intro' && <IntroScreen onStart={() => setScreen('consent')} />}
    {screen === 'consent' && <ConsentScreen
      onAgree={() => { setHasConsent(true); setScreen('selection'); }}
      onExit={() => { setHasConsent(false); setScreen('declined'); }}
    />}
    {screen === 'declined' && <DeclinedScreen onReturn={() => setScreen('intro')} />}
    {screen === 'selection' && <TrainingSelection {...{ selectedEmotions, toggleEmotion }} onContinue={() => setScreen('trainingIntro')} />}
    {screen === 'trainingIntro' && <InstructionScreen
      eyebrow="stage two / train the model"
      heading="Train the model"
      text="Labeled facial-landmark measurements from each participant are added to a shared in-memory training set for the local classifier. They stay in this page across participants until it is refreshed or closed; saved snapshots are separate photos for your strip."
      steps={[
        'You’ll be prompted to show different expressions. Follow the prompt and hold your expression during capture.',
        'Training capture begins when the countdown reaches 2. Hold your expression until capture finishes.',
        'FaceMesh measurements from the live frames train the model; one separate still is saved at the end of each countdown.'
      ]}
      button="I’m ready — start training"
      onClick={() => setScreen('training')}
    />}
    {screen === 'training' && <TrainingBooth {...{
      selectedEmotions, toggleEmotion, currentEmotion, trainingIndex, trainingPhotos,
      captureTraining, runCountdown, captureBusy, countdown, flash, cameraState,
      cameraError, startCamera, videoRef, enterMemeBooth, notice, modelReady,
      faceMeshReady, trainingCaptureStatus, onReviewInstructions: () => setScreen('trainingIntro'), reset
    }} />}
    {screen === 'trainingComplete' && <TransitionScreen heading="Training Complete" text="You've shown the AI how you express four different emotions. Now let's see how well it can read you without being told what emotion you're making." button="Enter the Meme Booth" onClick={enterMemeBooth} photos={trainingPhotos} onDownload={() => downloadStrip(trainingPhotos, false)} />}
    {screen === 'memeIntro' && <InstructionScreen
      eyebrow="stage three / try the meme booth"
      heading="Try the meme booth"
      text="The trained model interprets visible expressions and matches them with local memes; its reading may differ from how you actually feel."
      steps={[
        'Make an expression and watch the matching meme change beside your live camera view.',
        'Adjust your expression to see how the model responds.',
        'When you’re happy with the result, select “Take photo.”'
      ]}
      button="Start Meme Booth"
      onClick={() => setScreen('meme')}
    />}
    {screen === 'meme' && <MemeBooth {...{
      memePhotos, captureMeme, captureBusy, flash, cameraState, cameraError,
      startCamera, videoRef, memeNotice, faceMeshReady, faceDetected,
      currentPredictedEmotion, currentDisplayedMeme, loadedMeme,
      handleLiveMemeLoad, handleLiveMemeError,
      onReviewInstructions: () => setScreen('memeIntro')
    }} />}
    {screen === 'evaluate' && <EvaluationScreen item={evaluations[evaluationIndex]} index={evaluationIndex} onEvaluate={recordEvaluation} />}
    {screen === 'results' && <ResultsScreen {...{ memePhotos, evaluations, sessionData, reset }} />}
    <footer><span><Users size={14} /> {trainingPhotos.length}/4 photos · {trainingData.length} shared measurements</span><span>Predictions are interpretations of expressions, not feelings.</span></footer>
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
      {countdown && <div className="countdown" role="status" aria-live="polite"><span>{countdown}</span><small>hold that pose</small></div>}
      {flash && <div className="flash" />}{children}
    </div>
    <div className="camera-base"><span className="live-dot" /> live mirror <span><LockKeyhole size={12} /> no app upload</span></div>
  </div>;
}

function JourneyProgress({ currentStage }) {
  return <nav className="journey-progress" aria-label="Photobooth progress">
    <ol>
      {JOURNEY_STAGES.map((stage, index) => (
        <li
          key={stage}
          className={index < currentStage ? 'complete' : index === currentStage ? 'current' : ''}
          aria-current={index === currentStage ? 'step' : undefined}
        >
          <span className="journey-step-number">{String(index + 1).padStart(2, '0')}</span>
          <span>{stage}</span>
        </li>
      ))}
    </ol>
  </nav>;
}

function IntroScreen({ onStart }) {
  return <section className="instruction-card intro-card">
    <div className="intro-copy">
      <div className="eyebrow"><span className="eyebrow-dot" /> a tiny expression experiment</div>
      <h1>Welcome to <em>Purrception!</em></h1>
      <p className="intro-tagline">Can AI read your expressions?</p>
      <p>First, help train our little model by showing it different expressions. Then see which cat meme it picks for you.</p>
      <button className="button yellow" onClick={onStart}>Start Training <ChevronRight size={17} /></button>
    </div>
    <CatDoodle className="intro-cat" />
  </section>;
}

function ConsentScreen({ onAgree, onExit }) {
  const [checked, setChecked] = useState(false);

  return <section className="instruction-card consent-card">
    <div className="eyebrow"><span className="eyebrow-dot" /> required consent / before camera access</div>
    <h1>Before you begin: <em>image capture and model training</em></h1>
    <p className="consent-lead">During the first training step, the camera will capture images of you as you follow the expression prompts. These images will be used to train our model to associate your facial expressions with the prompted labels.</p>
    <p className="consent-model-note"><strong>How this model actually learns:</strong> FaceMesh processes live camera frames into facial-landmark measurements, and those labeled measurements train the classifier. The final still image is saved for your photo strip; its pixels are not model input.</p>
    <div className="consent-details">
      <h2>Data, access and deletion</h2>
      <ul>
        <li><strong>Processing and storage:</strong> Camera frames and captured photos are handled in this browser. Labeled facial-landmark measurements are retained in a shared training set in page memory and used to build the local classifier. They remain when starting another participant. The app has no upload endpoint or persistent browser storage; refreshing or closing this page clears the in-memory data. Still-photo pixels are not classifier input.</li>
        <li><strong>Who can access it:</strong> The open page and people with access to this browser or device. This project has no participant accounts or app backend. A downloaded photo strip is a file on your device.</li>
        <li><strong>When it is cleared:</strong> “Start another participant” clears the current participant’s photos and results but keeps the shared training measurements. Refresh or close the page to clear all in-memory data. The app does not delete photo strips you download.</li>
      </ul>
      <p className="consent-unknown"><strong>Still to confirm before public use:</strong> The project loads Google Fonts and ml5/MediaPipe dependencies. The repository does not establish what network or hosting logs those providers retain, or who can access them. No participant images or landmark measurements are sent by the app to an application server.</p>
    </div>
    <ol className="instruction-steps consent-steps">
      <li>Review how the camera, images and training measurements are handled.</li>
      <li>Choose whether to consent. The camera will not start until you agree and choose to start training.</li>
      <li>If you do not agree, exit without camera access or data collection.</li>
    </ol>
    <label className="consent-checkbox">
      <input type="checkbox" checked={checked} onChange={event => setChecked(event.target.checked)} />
      <span>I consent to my images being captured during training and used to train this project’s model.</span>
    </label>
    <div className="consent-actions">
      <button className="button yellow" onClick={onAgree} disabled={!checked}>I agree — continue to training</button>
      <button className="button cream dark-text" onClick={onExit}>Exit without participating</button>
    </div>
  </section>;
}

function DeclinedScreen({ onReturn }) {
  return <section className="instruction-card">
    <div className="eyebrow"><span className="eyebrow-dot" /> participation ended</div>
    <h1>You’re all set to <em>leave.</em></h1>
    <p>You declined consent. The camera was not started, and this visit did not collect training images or facial measurements.</p>
    <button className="button cream dark-text" onClick={onReturn}>Return to welcome</button>
  </section>;
}

function TrainingSelection({ selectedEmotions, toggleEmotion, onContinue }) {
  return <section className="booth-page">
    <div className="booth-intro"><div className="eyebrow"><span className="eyebrow-dot" /> stage two / train the model</div><h1>Choose your <em>emotions.</em></h1><p>Choose four labels and add your measurements to the shared training set from previous participants.</p></div>
    <ol className="instruction-steps selection-steps">
      <li>Choose four different expression labels.</li>
      <li>Follow each matching prompt during training.</li>
      <li>The labeled measurements train the model; the photos are saved separately for your strip.</li>
    </ol>
    <EmotionPicker {...{ selectedEmotions, toggleEmotion }} />
    <div className="selection-footer"><b>{selectedEmotions.length}/4 selected</b><span>Everyone expresses emotions differently. Choose four emotions and show the AI what they look like on you.</span>{selectedEmotions.length === TRAINING_COUNT && <button className="button yellow" onClick={onContinue}>Continue to training instructions <ChevronRight size={17} /></button>}</div>
  </section>;
}

function InstructionScreen({ eyebrow, heading, text, detail, steps = [], button, onClick }) {
  return <section className="instruction-card"><div className="eyebrow"><span className="eyebrow-dot purple" /> {eyebrow}</div><h1>{heading}</h1><p>{text}</p>{detail && <strong className="instruction-detail">{detail}</strong>}{steps.length > 0 && <ol className="instruction-steps">{steps.map(step => <li key={step}>{step}</li>)}</ol>}<button className="button yellow" onClick={onClick}>{button} <ChevronRight size={17} /></button></section>;
}

function TransitionScreen({ heading, text, button, onClick, photos, onDownload }) {
  return <section className="instruction-card"><div className="eyebrow"><span className="eyebrow-dot" /> your training photos are ready</div><h1>{heading}</h1><p>{text}</p><div className="transition-actions"><button className="button dark" onClick={onClick}>{button} <ChevronRight size={17} /></button><button className="button cream dark-text" onClick={onDownload}><Download size={17} /> Download Training Strip</button></div><PhotoStrip photos={photos} type="training" /></section>;
}

function EvaluationScreen({ item, index, onEvaluate }) {
  return <section className="evaluation-card"><div className="eyebrow"><span className="eyebrow-dot purple" /> evaluate the interpretation</div><h1>AI read your expression as: <em>{item?.label}</em></h1><p className="evaluation-progress">{index + 1} of 4 evaluated</p><div className="evaluation-pair"><img src={item?.photo} alt={`Participant photo ${index + 1}`} /><img src={item?.meme} alt={`${item?.label} cat meme`} /></div><h2>Did the AI read your expression the way you intended?</h2><div className="evaluation-actions"><button className="button yes" onClick={() => onEvaluate(true)}>👍 Yes</button><button className="button no" onClick={() => onEvaluate(false)}>👎 No</button></div></section>;
}

function ResultsScreen({ memePhotos, evaluations, reset }) {
  const agreementCount = evaluations.filter(item => item.accurate).length;
  return <section className="results-screen">
    <div className="booth-intro"><div className="eyebrow"><span className="eyebrow-dot" /> stage four / download</div><h1>The AI got <em>{agreementCount} out of 4</em> right.</h1><p>You agreed with the AI {agreementCount * 25}% of the time.</p></div>
    <div className="download-guidance">
      <h2>Download your photo strip</h2>
      <p>Keep a copy of your four captured photos and their matching memes.</p>
      <ol className="instruction-steps">
        <li>Review the saved photo-and-meme pairs.</li>
        <li>Download the strip to this device if you want to keep it.</li>
        <li>Choose “Start another participant” to clear your photos and results. Your labeled facial measurements stay in the shared training set for the next participant until this page is refreshed or closed.</li>
      </ol>
    </div>
    <PhotoStrip photos={memePhotos} type="meme" />
    <p className="closing-message"><b>Thanks for training the AI.</b><br />Your expressions helped teach the system, and your feedback helped evaluate its interpretations.</p>
    <div className="strip-actions"><button className="button dark" onClick={() => downloadStrip(memePhotos)}><Download size={17} /> Download photo strip</button><button className="button cream dark-text" onClick={reset}><RefreshCw size={16} /> Start another participant</button></div>
  </section>;
}

function TrainingBooth(props) {
  const {
    selectedEmotions, toggleEmotion, currentEmotion, trainingIndex, trainingPhotos,
    runCountdown, captureBusy, countdown, flash, cameraState, cameraError,
    startCamera, videoRef, enterMemeBooth, notice, modelReady, faceMeshReady,
    trainingCaptureStatus, onReviewInstructions, reset
  } = props;
  const choosing = selectedEmotions.length < TRAINING_COUNT;
  const complete = trainingPhotos.length === TRAINING_COUNT;
  const statusLabel = trainingCaptureStatus === 'capturing'
    ? 'Capturing training examples'
    : trainingCaptureStatus === 'complete' ? 'Capture complete' : 'Get ready';
  const retrying = Boolean(notice && !complete);
  return <section className="booth-page">
    <div className="booth-intro"><div className="eyebrow"><span className="eyebrow-dot" /> stage two / train the model</div><h1>Teach it <em>your face.</em></h1><p>Follow each expression prompt while the model learns from labeled facial measurements.</p></div>
    {choosing ? <EmotionPicker {...{ selectedEmotions, toggleEmotion }} /> : <div className="booth-layout">
      <div><BoothCamera {...props} theme="training-curtain"><div className="booth-prompt"><span>training photo {trainingIndex + 1} of 4</span><strong>{complete ? 'Training complete!' : EMOTION_META[currentEmotion].prompt}</strong><small>{complete ? 'Your four training photos are ready.' : 'One photo per label. If tracking misses, try again.'}</small></div></BoothCamera>
        <div className="training-guidance" aria-live="polite">
          <strong className={`training-status ${trainingCaptureStatus}`}>{statusLabel}</strong>
          <p><b>Current prompt:</b> {complete ? 'Training complete.' : EMOTION_META[currentEmotion].prompt}</p>
          <p>FaceMesh measurements from live frames during the final 2 seconds train the classifier. One separate still photo is captured at the end for your labeled photo strip; it is not model input.</p>
        </div>
        <div className="booth-controls">
          <div className="progress-dots">{selectedEmotions.map((label, index) => <i key={label} className={index < trainingIndex ? 'on' : ''} style={{ '--dot': EMOTION_META[label].color }} />)}</div>
          {!complete
            ? <div className="training-actions">
              <button className="button yellow wide" onClick={() => runCountdown(() => props.captureTraining?.())} disabled={captureBusy || countdown || !faceMeshReady}>
                <Camera size={18} /> {captureBusy ? 'Saving example…' : retrying ? 'Retry capture' : faceMeshReady ? `Capture ${currentEmotion}` : 'Starting face tracking…'}
              </button>
              <button className="button cream dark-text wide" onClick={onReviewInstructions}>Review training instructions</button>
            </div>
            : <button className="button yellow wide" onClick={enterMemeBooth}><ChevronRight size={17} /> Enter Meme Booth</button>}
        </div>
      </div>
      <div className="strip-column">{complete ? <PhotoStrip photos={trainingPhotos} type="training" /> : <EmptyStrip type="training" photos={trainingPhotos} />}<h2 className="thank-you">{complete ? 'Your expressions are trained.' : `${trainingPhotos.length} of 4 training photos captured`}</h2>{complete && <div className="strip-actions"><button className="button dark" onClick={enterMemeBooth}>Start posing freely <ChevronRight size={17} /></button><button className="button cream dark-text" onClick={reset}><RefreshCw size={16} /> Start another participant</button></div>}</div>
    </div>}
    {choosing && <div className="selection-footer"><b>{selectedEmotions.length} of 4 emotions selected</b><span>Pick four different labels to unlock the training camera.</span></div>}
    {notice && <p className="notice" role="status" aria-live="polite">{notice}</p>}
    {modelReady && <div className="model-panel"><div><b>Shared model updated</b><p>Your labeled measurements have been added to the shared training set. The Meme Booth compares new expressions with the accumulated examples.</p></div><button className="button dark" onClick={enterMemeBooth}>Continue <ChevronRight size={16} /></button></div>}
  </section>;
}

function EmotionPicker({ selectedEmotions, toggleEmotion }) {
  return <div className="emotion-picker"><div className="picker-heading"><div><b>Choose your four training emotions</b><p>These are the faces the booth will learn from you.</p></div><span>{selectedEmotions.length}/4</span></div><div className="emotion-grid">{EMOTIONS.map(emotion => {
    const selected = selectedEmotions.includes(emotion);
    return <button key={emotion} className={`emotion-option ${selected ? 'selected' : ''}`} onClick={() => toggleEmotion(emotion)} disabled={!selected && selectedEmotions.length >= TRAINING_COUNT} style={{ '--emotion-color': EMOTION_META[emotion].color }}><span className="emotion-emoji">{EMOTION_META[emotion].emoji}</span><span>{emotion}</span>{selected && <Check size={16} />}</button>;
  })}</div></div>;
}

function MemeBooth(props) {
  const {
    memePhotos, captureMeme, captureBusy, flash, cameraState, cameraError, startCamera,
    videoRef, memeNotice, faceMeshReady, faceDetected, currentPredictedEmotion,
    currentDisplayedMeme, loadedMeme, handleLiveMemeLoad, handleLiveMemeError,
    onReviewInstructions
  } = props;
  const canCapture = faceMeshReady
    && cameraState === 'ready'
    && faceDetected
    && Boolean(currentPredictedEmotion)
    && Boolean(currentDisplayedMeme)
    && loadedMeme === currentDisplayedMeme;
  const readingMessage = !faceDetected
    ? 'Move into frame so the AI can read your expression.'
    : 'Reading your expression…';
  const retryingCapture = memeNotice.toLowerCase().includes('photo could not be captured');

  return <section className="booth-page">
    <div className="booth-intro"><div className="eyebrow"><span className="eyebrow-dot purple" /> stage three / try the meme booth</div><h1>Pose <em>your way.</em></h1><p>Change your expression and watch the matching cat meme update. Take a photo when you like the pairing.</p></div>
    <div className="meme-live-layout">
      <div className="meme-live-camera">
        <BoothCamera {...{ videoRef, cameraState, cameraError, startCamera, flash }} theme="meme-curtain" />
        <p className="meme-panel-label">Live camera</p>
      </div>
      <section className="meme-live-result" aria-live="polite">
        <div className="meme-live-panel">
          {currentDisplayedMeme && <img
            key={currentDisplayedMeme}
            src={currentDisplayedMeme}
            alt={`${currentPredictedEmotion} cat meme`}
            onLoad={() => handleLiveMemeLoad(currentDisplayedMeme)}
            onError={handleLiveMemeError}
          />}
          {!currentDisplayedMeme && <p className="meme-live-placeholder">{readingMessage}</p>}
        </div>
        <p className="meme-panel-label">
          {currentPredictedEmotion
            ? `AI reads your expression as: ${currentPredictedEmotion.toUpperCase()}`
            : readingMessage}
        </p>
      </section>
    </div>
    <div className="meme-guidance">
      <p>Make an expression and watch the matching meme change beside your live camera view. Adjust your expression to see how the model responds. When you’re happy with the result, select <strong>Take photo</strong>.</p>
      <p>The result is the model’s interpretation of visible expressions, which may differ from how you actually feel.</p>
    </div>
    <div className="meme-booth-controls">
      <div className="progress-dots">{[0, 1, 2, 3].map(index =>
        <i key={index} className={index < memePhotos.length ? 'on' : ''} style={{ '--dot': '#8e72c5' }} />
      )}</div>
      <span className="meme-progress">{memePhotos.length} / {MEME_COUNT} photos captured</span>
      <div className="meme-action-buttons">
        <button className="button cream dark-text" onClick={onReviewInstructions}>Review Meme Booth instructions</button>
        <button className="button coral" onClick={captureMeme} disabled={captureBusy || !canCapture}>
          <Camera size={18} /> {captureBusy ? 'Saving photo…' : retryingCapture ? 'Retry photo' : 'Take Photo'}
        </button>
      </div>
    </div>
    {memeNotice && <p className="notice" role="status" aria-live="polite">{memeNotice}</p>}
  </section>;
}

function EmptyStrip({ type, photos = [] }) {
  return <div className={`empty-strip ${type}`}><div className="strip-top"><CatDoodle className="strip-cat" /> {type === 'meme' ? 'meme strip preview' : 'training photo strip'}</div>{[0, 1, 2, 3].map(index => <div className="empty-row" key={index}><span>{index + 1}</span><i>{photos[index] ? 'matched' : type === 'meme' ? 'pose + cat' : 'waiting for a face'}</i></div>)}<div className="strip-bottom">purrception</div></div>;
}

function PhotoStrip({ photos, type }) {
  if (!photos.length) return <EmptyStrip type={type} />;
  return <div className={`photo-strip ${type}`}><div className="strip-top"><CatDoodle className="strip-cat" /> {type === 'meme' ? 'purrception' : 'training photo strip'}</div>{photos.map((item, index) => <div className="strip-row" key={`${item.label}-${index}`}><div className="strip-photo"><img src={item.photo} alt={`Participant ${index + 1}`} /><small>{index + 1} · {item.label}</small></div>{type === 'meme' && <div className="strip-meme"><img src={item.meme} alt={`${item.label} cat meme`} /><small>{item.label} match</small></div>}</div>)}<div className="strip-bottom">local, playful, yours</div></div>;
}

createRoot(document.getElementById('root')).render(<App />);

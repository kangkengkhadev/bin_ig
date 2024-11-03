import React, { useRef, useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import * as poseDetection from '@tensorflow-models/pose-detection';

const PoseEstimationGame = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [detector, setDetector] = useState(null);
  const [isVideoReady, setIsVideoReady] = useState(false);
  const [status, setStatus] = useState("Loading...");
  const [score, setScore] = useState(0);
  const [gameOver, setGameOver] = useState(false);
  const [currentItem, setCurrentItem] = useState(null);
  const [timer, setTimer] = useState(15); // Timer state

  // Define the bins (static buttons)
  const bins = [
    { type: "general", name: "General wet", color: "#4B5563", image: "https://hospitalitythailand.com/uploads/202206/6297137393a86.png" },
    { type: "recycle", name: "Recyclable", color: "#3B82F6", image: "https://mw.co.th/uploads/202206/62970a115dc8e.png" },
    { type: "wet", name: "wet", color: "#10B981", image: "https://mw.co.th/uploads/202206/629713a699e2e.png" },
    { type: "hazardous", name: "Hazardous", color: "#EF4444", image: "https://hospitalitythailand.com/uploads/202206/629708034e62a.png" }
  ];

  // Items that can appear above player's head
  const items = [
    { name: "Plastic Bottle", type: "recycle", image: "https://img.lovepik.com/png/20230930/mineral-water-water-bottle-recover-drink_36069_wh860.png" },
    { name: "Banana Peel", type: "wet", image: "https://png.pngtree.com/png-clipart/20220108/ourmid/pngtree-banana-peel-decorative-pattern-illustration-png-image_4101651.png" },
    { name: "Battery", type: "hazardous", image: "https://e7.pngegg.com/pngimages/636/772/png-clipart-battery-battery-thumbnail.png" },
    { name: "Paper", type: "recycle", image: "https://png.pngtree.com/png-clipart/20220720/original/pngtree-toilet-tissue-paper-roll-vector-illustration-png-image_8388391.png" },
    { name: "Fishbone", type: "wet", image: "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR8Pc6CqNsOt25cGt3g3k8iIvHyIeX1RnpjFQ&s" },
    { name: "Candy Wrapper", type: "general", image: "https://t4.ftcdn.net/jpg/03/04/69/69/360_F_304696907_czmMiRwezOOmR4F3M4soUUfRSmiC7O2a.jpg" },
  ];

  const confidenceThreshold = 0.3;

  const initializeGame = () => {
    const randomIndex = Math.floor(Math.random() * items.length);
    const selected = items[randomIndex];
    setCurrentItem(selected);
    setGameOver(false);
    setTimer(15); // Reset timer to 15 seconds
    loadImage(selected.image);
  };

  const loadImage = (url) => {
    const img = new Image();
    img.src = url;
    img.onload = () => {
      setCurrentItem((prev) => ({ ...prev, element: img }));
    };
  };

  const handleAnswer = (binType) => {
    if (binType === currentItem.type) {
      setScore(prev => prev + 1);
      // Only generate a new item without resetting the timer
      const randomIndex = Math.floor(Math.random() * items.length);
      const selected = items[randomIndex];
      loadImage(selected.image);
      setCurrentItem(selected);
    } else {
      setScore(prev => Math.max(prev - 1, 0)); // Decrement score by 1 but don't let it go below 0
    }
  };

  const handleRestart = () => {
    setScore(0);
    initializeGame();
  };

  // Timer effect
  useEffect(() => {
    let timerInterval = null;
    if (!gameOver && timer > 0) {
      timerInterval = setInterval(() => {
        setTimer(prev => prev - 1);
      }, 1000);
    } else if (timer === 0) {
      setScore(prev => Math.max(prev - 1, 0)); // Decrement score by 1 when timer runs out
      setGameOver(true);
    }
    return () => clearInterval(timerInterval);
  }, [timer, gameOver]);

  // Model loading useEffect
  useEffect(() => {
    const loadModel = async () => {
      setStatus("Loading model...");
      await tf.ready();
      await tf.setBackend('cpu');

      const model = poseDetection.SupportedModels.MoveNet;
      const detectorConfig = { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }; // Using MoveNet's SINGLEPOSE_LIGHTNING model
      const newDetector = await poseDetection.createDetector(model, detectorConfig);
      setDetector(newDetector);
      setStatus("Model loaded");
      initializeGame();
    };
    loadModel();
  }, []);

  // Video initialization useEffect
  useEffect(() => {
    const startVideo = async () => {
      setStatus("Initializing video...");
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;

      videoRef.current.onloadedmetadata = () => {
        videoRef.current.play();
        setIsVideoReady(true);
        setStatus("Sort the waste!");

        canvasRef.current.width = videoRef.current.videoWidth;
        canvasRef.current.height = videoRef.current.videoHeight;
      };
    };
    startVideo();
  }, []);

  // Pose estimation useEffect
  useEffect(() => {
    if (detector && isVideoReady && currentItem?.element) {
      const detectPose = async () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        const drawSkeleton = (keypoints) => {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          const nose = keypoints.find((keypoint) => keypoint.name === 'nose' && keypoint.score > confidenceThreshold);
          if (nose) {
            const imageSize = 150;
            const yOffset = 150;
            ctx.drawImage(currentItem.element, nose.x - imageSize / 2, nose.y - imageSize / 2 - yOffset, imageSize, imageSize);
          }

          keypoints.forEach((keypoint) => {
            if (keypoint.score > confidenceThreshold) {
              ctx.beginPath();
              ctx.arc(keypoint.x, keypoint.y, 5, 0, 2 * Math.PI);
              ctx.fillStyle = 'red';
              ctx.fill();
            }
          });

          const connections = poseDetection.util.getAdjacentPairs(poseDetection.SupportedModels.MoveNet);
          connections.forEach(([i, j]) => {
            const kp1 = keypoints[i];
            const kp2 = keypoints[j];

            if (kp1.score > confidenceThreshold && kp2.score > confidenceThreshold) {
              ctx.beginPath();
              ctx.moveTo(kp1.x, kp1.y);
              ctx.lineTo(kp2.x, kp2.y);
              ctx.lineWidth = 2;
              ctx.strokeStyle = 'blue';
              ctx.stroke();
            }
          });
        };

        const detectAndDraw = async () => {
          if (video.videoWidth > 0 && video.videoHeight > 0) {
            const poses = await detector.estimatePoses(video);
            if (poses.length > 0) {
              drawSkeleton(poses[0].keypoints);
            }
          }
          requestAnimationFrame(detectAndDraw);
        };
        detectAndDraw();
      };
      detectPose();
    }
  }, [detector, isVideoReady, currentItem]);

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      minHeight: '100vh',
      backgroundColor: '#f0f2f5',
      padding: '1rem'
    }}>
      {/* Score Display */}
      <div style={{
        position: 'fixed',
        top: '1rem',
        right: '1rem',
        backgroundColor: '#ffffff',
        padding: '0.5rem 1rem',
        borderRadius: '2rem',
        boxShadow: '0 2px 10px rgba(0, 0, 0, 0.1)',
        fontSize: '1.5rem'
      }}>
        Score: {score} | Time: {timer} seconds
      </div>

      {/* Video and Canvas */}
      <video ref={videoRef} style={{ display: 'none' }} />
      <canvas ref={canvasRef} />

      {/* Current Item Display */}

      {/* Bin Buttons */}
      <div style={{ display: 'flex', justifyContent: 'space-around', marginTop: '2rem' }}>
  {bins.map((bin) => (
    <button
      key={bin.type}
      onClick={() => handleAnswer(bin.type)}
      style={{
        color: '#ffffff',
        border: 'none',
        borderRadius: '1rem',
        padding: '1rem',
        cursor: 'pointer',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center' // Center align items in the button
      }}
    >
      <img src={bin.image} alt={bin.name} style={{ width: '70px', height: '70px', marginBottom: '0.5rem' }} />
    </button>
  ))}
</div>

      {/* Restart Button */}
      {gameOver && (
        <button onClick={handleRestart} style={{
          marginTop: '2rem',
          padding: '1rem 2rem',
          border: 'none',
          borderRadius: '1rem',
          backgroundColor: '#3B82F6',
          color: '#ffffff',
          cursor: 'pointer'
        }}>
          Restart Game
        </button>
      )}
    </div>
  );
};

export default PoseEstimationGame;

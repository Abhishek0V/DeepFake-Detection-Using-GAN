import { useRef, useCallback } from 'react';

// WAV encoding utilities
function flattenBuffer(channelBuffer, recordingLength) {
  const result = new Float32Array(recordingLength);
  let offset = 0;
  for (let i = 0; i < channelBuffer.length; i++) {
    result.set(channelBuffer[i], offset);
    offset += channelBuffer[i].length;
  }
  return result;
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

function floatTo16BitPCM(output, offset, input) {
  for (let i = 0; i < input.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, input[i]));
    output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
}

function bufferToWav(buffer, sampleRate) {
  const bufferLength = buffer.length;
  const wavBuffer = new ArrayBuffer(44 + bufferLength * 2);
  const view = new DataView(wavBuffer);

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + bufferLength * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, bufferLength * 2, true);
  floatTo16BitPCM(view, 44, buffer);

  return new Blob([view], { type: 'audio/wav' });
}

const REC_SAMPLE_RATE = 16000;

/**
 * useRecorder – custom hook encapsulating all Web Audio API recording logic.
 * Returns { isRecording, startRecording, stopRecording, canvasRef, timerText }
 */
export function useRecorder({ onFileReady }) {
  const isRecordingRef = useRef(false);
  const audioContextRef = useRef(null);
  const recordingStreamRef = useRef(null);
  const animationFrameRef = useRef(null);
  const recordTimerRef = useRef(null);
  const leftChannelRef = useRef([]);
  const recordingLengthRef = useRef(0);
  const canvasRef = useRef(null);
  const timerElRef = useRef(null);
  const recordStartTimeRef = useRef(null);

  // Draw waveform on canvas
  const visualize = useCallback((analyser) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.parentElement?.clientWidth || 300;
    canvas.height = 80;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      if (!isRecordingRef.current) return;
      animationFrameRef.current = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArray);

      ctx.fillStyle = 'rgba(10, 14, 26, 0.4)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.lineWidth = 2;
      const grad = ctx.createLinearGradient(0, 0, canvas.width, 0);
      grad.addColorStop(0, '#6366f1');
      grad.addColorStop(1, '#8b5cf6');
      ctx.strokeStyle = grad;
      ctx.beginPath();

      const sliceWidth = (canvas.width * 1.0) / bufferLength;
      let x = 0;
      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
    };
    draw();
  }, []);

  const stopRecordingState = useCallback(() => {
    isRecordingRef.current = false;
    if (recordTimerRef.current) clearInterval(recordTimerRef.current);
    if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
    }
  }, []);

  const startRecording = useCallback(async (onStateChange) => {
    isRecordingRef.current = true;
    leftChannelRef.current = [];
    recordingLengthRef.current = 0;

    onStateChange?.('recording');

    try {
      recordingStreamRef.current = await navigator.mediaDevices.getUserMedia({
        audio: true,
        video: false,
      });

      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: REC_SAMPLE_RATE,
      });

      const source = audioContextRef.current.createMediaStreamSource(
        recordingStreamRef.current
      );
      const analyser = audioContextRef.current.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);

      const bufferSize = 2048;
      // eslint-disable-next-line no-undef
      const recorderNode = audioContextRef.current.createScriptProcessor(
        bufferSize, 1, 1
      );
      recorderNode.onaudioprocess = (e) => {
        if (!isRecordingRef.current) return;
        const left = e.inputBuffer.getChannelData(0);
        leftChannelRef.current.push(new Float32Array(left));
        recordingLengthRef.current += bufferSize;
      };

      source.connect(recorderNode);
      recorderNode.connect(audioContextRef.current.destination);

      visualize(analyser);

      // Timer
      recordStartTimeRef.current = Date.now();
      onStateChange?.('timer', '00:00');
      recordTimerRef.current = setInterval(() => {
        const elapsed = Date.now() - recordStartTimeRef.current;
        const seconds = Math.floor((elapsed / 1000) % 60);
        const minutes = Math.floor((elapsed / 1000 / 60) % 60);
        const timeStr =
          (minutes < 10 ? '0' + minutes : minutes) +
          ':' +
          (seconds < 10 ? '0' + seconds : seconds);
        onStateChange?.('timer', timeStr);
      }, 1000);
    } catch (err) {
      console.error('Microphone error:', err);
      alert('Could not access microphone: ' + err.message);
      stopRecordingState();
      onStateChange?.('idle');
    }
  }, [visualize, stopRecordingState]);

  const stopRecording = useCallback((onStateChange) => {
    if (recordingStreamRef.current) {
      recordingStreamRef.current.getTracks().forEach((t) => t.stop());
    }
    stopRecordingState();

    if (recordingLengthRef.current > 0) {
      const flat = flattenBuffer(
        leftChannelRef.current,
        recordingLengthRef.current
      );
      const wavBlob = bufferToWav(flat, REC_SAMPLE_RATE);
      const file = new File(
        [wavBlob],
        `recorded_audio_${Date.now()}.wav`,
        { type: 'audio/wav' }
      );
      onFileReady(file);
      onStateChange?.('saved');
    } else {
      onStateChange?.('idle');
    }
  }, [stopRecordingState, onFileReady]);

  return { canvasRef, startRecording, stopRecording };
}

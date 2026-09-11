import logging
from collections import deque
from typing import Optional
import numpy as np

logger = logging.getLogger("voicebot.vad")

class SileroVADDetector:
    """
    Voice Activity Detection using Silero VAD with rolling buffer,
    pre-speech buffering, and silence timeout to detect complete utterances.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        silence_duration_ms: int = 700,
        min_speech_duration_ms: int = 250,
        pre_speech_padding_ms: int = 250,
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.window_size_samples = 512 if sample_rate == 16000 else 256
        
        self.min_silence_samples = int((silence_duration_ms / 1000.0) * sample_rate)
        self.min_speech_samples = int((min_speech_duration_ms / 1000.0) * sample_rate)
        self.pre_speech_samples = int((pre_speech_padding_ms / 1000.0) * sample_rate)

        self.model = None
        self._init_model()

        # Barge-in configuration
        self.is_bot_speaking = False
        self.is_bot_processing = False
        self.barge_in_speech_samples = 0
        self.min_barge_in_samples = int(0.24 * sample_rate)  # 240ms (~1 syllable) to trigger barge-in

        # Noise resilience & Debouncing
        self.ambient_noise_rms = 0.005
        self.consecutive_speech_windows = 0
        self.voiced_samples_count = 0

        # Session state
        self.reset()

    def set_bot_state(self, is_speaking: bool, is_processing: bool = False):
        """Notifies VAD whether the bot is currently speaking or processing."""
        self.is_bot_speaking = is_speaking
        self.is_bot_processing = is_processing
        if not is_speaking and not is_processing:
            self.barge_in_speech_samples = 0

    def set_bargein_mode(self, is_bot_speaking: bool):
        """Standard Step 2 API: sets sensitivity mode for barge-in detection."""
        self.set_bot_state(is_speaking=is_bot_speaking, is_processing=False)

    def set_bot_speaking(self, is_speaking: bool):
        """Compatibility alias for set_bot_state."""
        self.set_bot_state(is_speaking=is_speaking, is_processing=False)

    def _init_model(self):
        """Loads Silero VAD model via faster_whisper ONNX or torch.hub, with fallback."""
        # 1. Try faster_whisper built-in ONNX Silero VAD (lightweight, ~1ms inference)
        try:
            from faster_whisper.vad import get_vad_model
            self.model = get_vad_model()
            self.is_onnx_model = True
            logger.info("Silero VAD ONNX model loaded successfully via faster_whisper.")
            return
        except Exception as e:
            logger.debug(f"Could not load faster_whisper ONNX VAD: {e}")

        # 2. Try torch.hub
        try:
            import torch
            self.torch = torch
            self.model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self.model.eval()
            self.is_onnx_model = False
            logger.info("Silero VAD model loaded successfully via torch.hub.")
            return
        except Exception as e:
            logger.warning(f"Failed to load Silero VAD via torch ({e}). Falling back to energy-based VAD.")
            self.model = None
            self.is_onnx_model = False

    def reset(self):
        """Resets the internal buffer state for a new session or utterance."""
        self.is_speaking = False
        self.speech_buffer = []
        self.silence_samples = 0
        self.remainder_buffer = np.array([], dtype=np.float32)
        self.barge_in_speech_samples = 0
        self.consecutive_silence_samples = 0
        self.consecutive_speech_windows = 0
        self.voiced_samples_count = 0

        # Stateful streaming Silero RNN states
        self._h = np.zeros((1, 1, 128), dtype="float32")
        self._c = np.zeros((1, 1, 128), dtype="float32")
        self._context = np.zeros((1, 64), dtype="float32")
        
        # Ring buffer for pre-speech audio frames
        max_pre_frames = max(1, self.pre_speech_samples // self.window_size_samples)
        self.pre_speech_buffer = deque(maxlen=max_pre_frames)

    def _calculate_chunk_prob(self, chunk: np.ndarray) -> float:
        """Calculates speech probability for a 512-sample float32 chunk using streaming states."""
        if self.model is not None:
            try:
                if getattr(self, "is_onnx_model", False):
                    # Fast streaming ONNX inference with persistent RNN states
                    inp = np.concatenate((self._context, chunk.reshape(1, -1)), axis=1).astype(np.float32)
                    self._context = chunk[-64:].reshape(1, 64).astype(np.float32)
                    out, self._h, self._c = self.model.session.run(
                        None,
                        {"input": inp, "h": self._h, "c": self._c}
                    )
                    return float(out[0])
                elif hasattr(self, "torch"):
                    tensor_chunk = self.torch.from_numpy(chunk)
                    with self.torch.no_grad():
                        prob = self.model(tensor_chunk, self.sample_rate).item()
                    return prob
            except Exception as ex:
                logger.debug(f"Silero inference error ({ex}), using energy fallback.")

        # Energy-based fallback: compute RMS
        rms = np.sqrt(np.mean(chunk**2) + 1e-8)
        return 0.85 if rms > 0.015 else 0.10

    def process_chunk(self, audio_chunk: np.ndarray) -> tuple[bool, Optional[np.ndarray], bool]:
        """
        Processes an incoming audio chunk (1D float32 numpy array, values [-1, 1]).
        Returns:
            (is_currently_speaking: bool, complete_utterance_audio: Optional[np.ndarray], barge_in_detected: bool)
        """
        # Combine remainder with new chunk
        if len(self.remainder_buffer) > 0:
            audio_data = np.concatenate((self.remainder_buffer, audio_chunk))
        else:
            audio_data = audio_chunk

        total_samples = len(audio_data)
        utterance_completed = None
        barge_in_detected = False
        current_idx = 0

        # Adjust sensitivity based on bot state (Acoustic Echo suppression & Barge-in)
        if self.is_bot_speaking:
            threshold = self.threshold + 0.05  # 0.55 threshold (sensitive enough for natural human voice to barge-in)
            min_rms = max(0.010, self.ambient_noise_rms * 2.0)
            barge_in_target = self.min_barge_in_samples  # 240ms of clear voice to interrupt
        elif self.is_bot_processing:
            threshold = self.threshold  # 0.50 standard threshold
            min_rms = max(0.008, self.ambient_noise_rms * 1.8)
            barge_in_target = self.min_barge_in_samples  # 240ms to cancel background thinking
        else:
            threshold = self.threshold
            min_rms = max(0.008, self.ambient_noise_rms * 1.8)
            barge_in_target = 0

        while current_idx + self.window_size_samples <= total_samples:
            window = audio_data[current_idx : current_idx + self.window_size_samples]
            current_idx += self.window_size_samples

            rms = np.sqrt(np.mean(window**2) + 1e-8)
            prob = self._calculate_chunk_prob(window)

            # Adapt background noise floor during quiet moments
            if not self.is_speaking and prob < 0.25:
                self.ambient_noise_rms = float(np.clip(0.95 * self.ambient_noise_rms + 0.05 * rms, 0.003, 0.040))

            is_speech = (prob >= threshold) and (rms >= min_rms)

            if is_speech:
                self.consecutive_speech_windows += 1

                # Require at least 2 consecutive speech windows (64ms) to debounce noise clicks/claps
                if not self.is_speaking and self.consecutive_speech_windows >= 2:
                    self.is_speaking = True
                    self.speech_buffer = []
                    self.voiced_samples_count = 0
                    # Prepend pre-speech frames for smooth utterance onset
                    for pre_frame in self.pre_speech_buffer:
                        self.speech_buffer.append(pre_frame)
                    self.pre_speech_buffer.clear()
                    logger.debug("VAD: Confirmed speech onset (debounced).")

                if self.is_speaking:
                    self.speech_buffer.append(window)
                    self.voiced_samples_count += self.window_size_samples
                    self.silence_samples = 0
                    self.consecutive_silence_samples = 0

                    # Check Barge-in condition
                    if (self.is_bot_speaking or self.is_bot_processing) and barge_in_target > 0:
                        self.barge_in_speech_samples += self.window_size_samples
                        if self.barge_in_speech_samples >= barge_in_target and not barge_in_detected:
                            logger.info(
                                f"VAD: Barge-in triggered! User interrupted bot (speaking={self.is_bot_speaking}, processing={self.is_bot_processing}, samples={self.barge_in_speech_samples})"
                            )
                            barge_in_detected = True
                            self.barge_in_speech_samples = 0
                    else:
                        self.barge_in_speech_samples = 0

                # Prevent endless recording: cut off at 8 seconds max
                if self.is_speaking and len(self.speech_buffer) * self.window_size_samples >= 8 * self.sample_rate:
                    logger.info("VAD: Maximum speech duration (8s) reached. Forcing utterance completion.")
                    utterance_completed = np.concatenate(self.speech_buffer)
                    self.is_speaking = False
                    self.speech_buffer = []
                    self.silence_samples = 0
                    self.barge_in_speech_samples = 0
                    self.voiced_samples_count = 0
                    self.consecutive_speech_windows = 0
            else:
                self.consecutive_speech_windows = 0
                self.consecutive_silence_samples += self.window_size_samples

                # Only decay barge-in accumulator if there is sustained silence (>160ms)
                if self.consecutive_silence_samples >= int(0.16 * self.sample_rate):
                    self.barge_in_speech_samples = max(0, self.barge_in_speech_samples - self.window_size_samples)

                if self.is_speaking:
                    # Voice was active, now silent
                    self.speech_buffer.append(window)
                    self.silence_samples += self.window_size_samples

                    if self.silence_samples >= self.min_silence_samples:
                        # Silence threshold exceeded -> Check if actual voiced duration meets minimum
                        logger.debug(
                            f"VAD: Utterance pause detected. Voiced duration: {self.voiced_samples_count/self.sample_rate:.2f}s."
                        )
                        if self.voiced_samples_count >= self.min_speech_samples:
                            utterance_completed = np.concatenate(self.speech_buffer)
                            logger.info(
                                f"VAD: Utterance completed successfully ({len(utterance_completed)/self.sample_rate:.2f}s total, {self.voiced_samples_count/self.sample_rate:.2f}s voiced)."
                            )
                        else:
                            logger.info(
                                f"VAD: Discarded transient noise (only {self.voiced_samples_count/self.sample_rate:.2f}s voiced, needed {self.min_speech_samples/self.sample_rate:.2f}s)."
                            )

                        # Reset state for next utterance
                        self.is_speaking = False
                        self.speech_buffer = []
                        self.silence_samples = 0
                        self.voiced_samples_count = 0
                        self.consecutive_speech_windows = 0
                else:
                    # Idle silence, keep in pre-speech rolling buffer
                    self.pre_speech_buffer.append(window)

        # Save remaining samples for next call
        self.remainder_buffer = audio_data[current_idx:]

        return self.is_speaking, utterance_completed, barge_in_detected

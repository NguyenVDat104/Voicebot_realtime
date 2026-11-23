// Vietnamese Real-Time Voicebot Client Logic
// Full-Duplex, Sentence-level Streaming, Barge-in, History Export, Auto-Reconnect & Audio Replay
// Upgraded: Human-Centered Design System & Smart UI Controls

class VoicebotClient {
  constructor() {
    this.ws = null;
    this.audioContext = null;
    this.mediaStream = null;
    this.scriptProcessor = null;
    this.analyser = null;
    this.isRecording = false;
    this.isBotSpeaking = false;
    this.sessionId = null;

    // Sentence-level TTS Audio Queue & streaming buffer
    this.currentSentenceChunks = [];
    this.playbackQueue = [];
    this.isPlayingQueue = false;
    this.allSentencesDone = false;
    this.createdObjectUrls = [];
    this.currentAudioPlayer = null;
    this.currentAudioUrl = null;

    // Auto-reconnect state
    this.isExplicitDisconnect = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectTimeout = null;

    // Push-to-Talk & User audio recording buffer
    this.isSpacePressed = false;
    this.userSpeechPcmChunks = [];

    // Conversation history
    this.chatHistory = [];

    // Visualizer animation time tracker
    this.visualizerTime = 0;

    // DOM Elements - Navigation & Status
    this.connectionBadge = document.getElementById("connectionBadge");
    this.connectionText = document.getElementById("connectionText");
    this.sessionDisplay = document.getElementById("sessionDisplay");
    this.currentModeTag = document.getElementById("currentModeTag");
    this.indicatorIcon = document.getElementById("indicatorIcon");
    this.indicatorText = document.getElementById("indicatorText");
    this.mainMicBtn = document.getElementById("mainMicBtn");
    this.micBtnHint = document.getElementById("micBtnHint");
    this.chatContainer = document.getElementById("chatContainer");
    this.canvas = document.getElementById("audioVisualizer");
    this.canvasCtx = this.canvas.getContext("2d");

    // DOM Elements - Action & Export Buttons
    this.clearChatBtn = document.getElementById("clearChatBtn");
    this.exportTxtBtn = document.getElementById("exportTxtBtn");
    this.exportJsonBtn = document.getElementById("exportJsonBtn");
    this.textPromptInput = document.getElementById("textPromptInput");
    this.sendTextBtn = document.getElementById("sendTextBtn");

    // DOM Elements - Settings Modal
    this.toggleSettingsBtn = document.getElementById("toggleSettingsBtn");
    this.settingsModal = document.getElementById("settingsModal");
    this.closeSettingsBtn = document.getElementById("closeSettingsBtn");
    this.saveSettingsBtn = document.getElementById("saveSettingsBtn");
    this.wsUrlInput = document.getElementById("wsUrlInput");
    this.sessionIdInput = document.getElementById("sessionIdInput");
    this.vadSensitivityInput = document.getElementById("vadSensitivity");
    this.vadThresholdInput = document.getElementById("vadThreshold");
    this.ttsVoiceSelect = document.getElementById("ttsVoiceSelect");
    this.ttsRateSelect = document.getElementById("ttsRateSelect");

    // DOM Elements - Latency Metrics
    this.sttMetricVal = document.getElementById("sttMetricVal");
    this.llmMetricVal = document.getElementById("llmMetricVal");
    this.ttsMetricVal = document.getElementById("ttsMetricVal");
    this.totalMetricVal = document.getElementById("totalMetricVal");
    this.totalMetricDetailVal = document.getElementById("totalMetricDetailVal");

    this.initSettings();
    this.loadChatHistory();
    this.initEventListeners();
    this.initVisualizer();
    this.initAuth();
    this.initTabNavigation();
    this.initSubpageActions();
  }

  getAssistantName() {
    return "Trợ lý AI";
  }

  initSettings() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const defaultWsUrl = `${protocol}//${window.location.host}/ws/audio`;
    this.wsUrl = localStorage.getItem("voicebot_ws_url") || defaultWsUrl;
    this.wsUrlInput.value = this.wsUrl;

    const savedSilence = localStorage.getItem("voicebot_vad_silence") || "600";
    const savedThreshold = localStorage.getItem("voicebot_vad_threshold") || "0.45";
    const savedVoice = localStorage.getItem("voicebot_tts_voice") || "vi-VN-HoaiMyNeural";
    const savedRate = localStorage.getItem("voicebot_tts_rate") || "+15%";

    if (this.vadSensitivityInput) this.vadSensitivityInput.value = savedSilence;
    if (this.vadThresholdInput) this.vadThresholdInput.value = savedThreshold;
    if (this.ttsVoiceSelect) this.ttsVoiceSelect.value = savedVoice;
    if (this.ttsRateSelect) this.ttsRateSelect.value = savedRate;

    // Sync visual cards and pills
    this.syncVoiceCards(savedVoice);
    this.syncSpeedPills(savedRate);
    this.syncEnvCards(savedThreshold);
  }

  syncVoiceCards(selectedVoice) {
    document.querySelectorAll(".voice-card").forEach((card) => {
      if (card.dataset.voice === selectedVoice) {
        card.classList.add("active");
      } else {
        card.classList.remove("active");
      }
    });
  }

  syncSpeedPills(selectedRate) {
    document.querySelectorAll(".speed-pill").forEach((pill) => {
      if (pill.dataset.rate === selectedRate) {
        pill.classList.add("active");
      } else {
        pill.classList.remove("active");
      }
    });
  }

  syncEnvCards(thresholdVal) {
    const thresh = parseFloat(thresholdVal);
    document.querySelectorAll(".env-card").forEach((card) => {
      const cardThresh = parseFloat(card.dataset.threshold);
      if (Math.abs(thresh - cardThresh) < 0.06) {
        card.classList.add("active");
      } else {
        card.classList.remove("active");
      }
    });
  }

  loadChatHistory() {
    try {
      const saved = localStorage.getItem("voicebot_chat_history");
      if (saved) {
        this.chatHistory = JSON.parse(saved);
        if (Array.isArray(this.chatHistory) && this.chatHistory.length > 0) {
          this.chatContainer.innerHTML = "";
          for (const item of this.chatHistory) {
            this.appendMessage(item.role, item.text, null, false, item.senderName);
          }
        }
      }
    } catch (e) {
      console.warn("Could not load chat history:", e);
      this.chatHistory = [];
    }
  }

  sendSettingsToServer() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      const silenceVal = this.vadSensitivityInput ? parseInt(this.vadSensitivityInput.value) : 600;
      const thresholdVal = this.vadThresholdInput ? parseFloat(this.vadThresholdInput.value) : 0.45;
      const voiceVal = this.ttsVoiceSelect ? this.ttsVoiceSelect.value : "vi-VN-HoaiMyNeural";
      const rateVal = this.ttsRateSelect ? this.ttsRateSelect.value : "+15%";

      this.ws.send(JSON.stringify({
        type: "update_settings",
        silence_duration_ms: silenceVal,
        threshold: thresholdVal,
        voice: voiceVal,
        rate: rateVal,
      }));
    }
  }

  showToast(message, type = "info", duration = 3500) {
    const container = document.getElementById("toastContainer");
    if (!container) return;

    const toast = document.createElement("div");
    toast.className = `toast-message toast-${type}`;
    const icon = type === "success" ? "✅" : type === "warning" ? "⚠️" : "✨";
    toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;

    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(10px) scale(0.95)";
      toast.style.transition = "all 0.25s ease";
      setTimeout(() => toast.remove(), 250);
    }, duration);
  }

  initEventListeners() {
    this.mainMicBtn.addEventListener("click", () => this.toggleConversation());
    this.clearChatBtn.addEventListener("click", () => this.clearChat());

    if (this.exportTxtBtn) {
      this.exportTxtBtn.addEventListener("click", () => this.exportAsText());
    }
    if (this.exportJsonBtn) {
      this.exportJsonBtn.addEventListener("click", () => this.exportAsJson());
    }

    // Hybrid Text Input
    if (this.sendTextBtn) {
      this.sendTextBtn.addEventListener("click", () => this.submitTextInput());
    }
    if (this.textPromptInput) {
      this.textPromptInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          this.submitTextInput();
        }
      });
    }

    // Quick Starter Prompt Chips
    document.querySelectorAll(".starter-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        const prompt = chip.dataset.prompt;
        if (prompt && this.textPromptInput) {
          this.textPromptInput.value = prompt;
          this.submitTextInput();
        }
      });
    });

    // Voice Card Selection
    document.querySelectorAll(".voice-card").forEach((card) => {
      card.addEventListener("click", () => {
        const voice = card.dataset.voice;
        this.ttsVoiceSelect.value = voice;
        this.syncVoiceCards(voice);
      });
    });

    // Speed Pills Selection
    document.querySelectorAll(".speed-pill").forEach((pill) => {
      pill.addEventListener("click", () => {
        const rate = pill.dataset.rate;
        this.ttsRateSelect.value = rate;
        this.syncSpeedPills(rate);
      });
    });

    // Environment Presets Selection
    document.querySelectorAll(".env-card").forEach((card) => {
      card.addEventListener("click", () => {
        const silence = card.dataset.silence;
        const threshold = card.dataset.threshold;
        if (this.vadSensitivityInput) this.vadSensitivityInput.value = silence;
        if (this.vadThresholdInput) this.vadThresholdInput.value = threshold;
        this.syncEnvCards(threshold);
      });
    });

    // Settings actions (Modal or Dedicated Tab)
    if (this.toggleSettingsBtn) {
      this.toggleSettingsBtn.addEventListener("click", () => {
        if (this.settingsModal) this.settingsModal.classList.remove("hidden");
      });
    }
    if (this.closeSettingsBtn) {
      this.closeSettingsBtn.addEventListener("click", () => {
        if (this.settingsModal) this.settingsModal.classList.add("hidden");
      });
    }
    if (this.settingsModal) {
      this.settingsModal.addEventListener("click", (e) => {
        if (e.target === this.settingsModal) {
          this.settingsModal.classList.add("hidden");
        }
      });
    }

    if (this.saveSettingsBtn) {
      this.saveSettingsBtn.addEventListener("click", () => this.saveSettings());
    }
    const saveMainBtn = document.getElementById("saveMainSettingsBtn");
    if (saveMainBtn) {
      saveMainBtn.addEventListener("click", () => this.saveSettings());
    }

    // Keyboard Shortcuts:
    // Esc: Interrupt bot speech
    // Space: Push-to-Talk (when not typing in an input field)
    window.addEventListener("keydown", async (e) => {
      if (e.key === "Escape" && this.isBotSpeaking) {
        this.triggerManualInterrupt();
      } else if (e.code === "Space" && document.activeElement !== this.textPromptInput && !e.repeat) {
        e.preventDefault();
        this.isSpacePressed = true;
        if (!this.isRecording) {
          await this.startConversation();
        }
        this.updateStatus("listening", "Đang giữ phím Space để nói...");
      }
    });

    window.addEventListener("keyup", (e) => {
      if (e.code === "Space" && this.isSpacePressed) {
        e.preventDefault();
        this.isSpacePressed = false;
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({ type: "force_utterance_complete" }));
        }
      }
    });

    // Window Resize Canvas
    window.addEventListener("resize", () => this.resizeCanvas());
    this.resizeCanvas();
  }

  async submitTextInput() {
    if (!this.textPromptInput) return;
    const text = this.textPromptInput.value.trim();
    if (!text) return;
    this.textPromptInput.value = "";

    try {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
        this.updateStatus("connecting", "Đang kết nối lại máy chủ...");
        await this.connectWebSocket();
      }

      this.stopCurrentAudio();
      this.appendMessage("user", text);
      this.showBotThinking();
      this.ws.send(JSON.stringify({
        type: "user_text",
        text: text
      }));
    } catch (err) {
      console.error("Error submitting text prompt:", err);
      this.showToast("Không thể gửi tin nhắn: " + err.message, "warning");
    }
  }

  triggerManualInterrupt() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "client_interrupt" }));
    }
    this.handleInterrupt("Phím tắt Esc");
  }

  resizeCanvas() {
    if (this.canvas && this.canvas.parentElement) {
      this.canvas.width = Math.min(this.canvas.parentElement.clientWidth, 400);
      this.canvas.height = 100;
    }
  }

  async toggleConversation() {
    if (this.isRecording) {
      this.stopConversation();
    } else {
      await this.startConversation();
    }
  }

  async startConversation() {
    try {
      this.isExplicitDisconnect = false;
      this.updateStatus("connecting", "Đang khởi tạo kết nối...");
      await this.connectWebSocket();
      await this.initMicrophone();

      this.isRecording = true;
      this.mainMicBtn.classList.add("recording");
      if (this.micBtnHint) this.micBtnHint.innerText = "Đang trò chuyện (Chạm để dừng)";
      this.updateStatus("listening", "Đang lắng nghe bạn nói...");
    } catch (err) {
      console.error("Error starting conversation:", err);
      alert("Không thể truy cập Microphone hoặc kết nối Server: " + err.message);
      this.stopConversation();
    }
  }

  stopConversation() {
    this.isExplicitDisconnect = true;
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    this.reconnectAttempts = 0;

    this.isRecording = false;
    this.mainMicBtn.classList.remove("recording");
    if (this.micBtnHint) this.micBtnHint.innerText = "Chạm để bắt đầu nói";

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }
    if (this.scriptProcessor) {
      this.scriptProcessor.disconnect();
      this.scriptProcessor = null;
    }
    if (this.audioContext && this.audioContext.state !== "closed") {
      this.audioContext.close();
      this.audioContext = null;
    }
    this.stopCurrentAudio();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.updateStatus("disconnected", "Nhấn nút gọi để bắt đầu trò chuyện");
  }

  connectWebSocket() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.wsUrl);
      this.ws.binaryType = "arraybuffer";

      this.ws.onopen = () => {
        console.log("WebSocket connected.");
        this.connectionBadge.classList.add("connected");
        this.connectionText.innerText = "Sẵn sàng";
        if (this.reconnectAttempts > 0) {
          this.showToast("Đã khôi phục kết nối thành công!", "success");
        }
        this.reconnectAttempts = 0;
        this.sendSettingsToServer();
        resolve();
      };

      this.ws.onclose = () => {
        console.log("WebSocket closed.");
        this.connectionBadge.classList.remove("connected");
        this.connectionText.innerText = "Đang ngắt kết nối";

        // Auto-reconnect with exponential backoff if not closed manually
        if (!this.isExplicitDisconnect && this.isRecording) {
          if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(1.8, this.reconnectAttempts - 1), 8000);
            this.showToast(`Mất kết nối. Đang thử lại (${this.reconnectAttempts}/${this.maxReconnectAttempts}) sau ${(delay / 1000).toFixed(1)}s...`, "warning", delay);
            this.reconnectTimeout = setTimeout(() => {
              this.connectWebSocket().then(() => this.sendSettingsToServer()).catch((e) => console.warn("Retry failed:", e));
            }, delay);
          } else {
            this.showToast("Không thể kết nối lại máy chủ. Vui lòng bấm mic để kết nối lại.", "warning", 6000);
            this.stopConversation();
          }
        } else if (this.isRecording) {
          this.stopConversation();
        }
      };

      this.ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        reject(err);
      };

      this.ws.onmessage = (event) => {
        if (typeof event.data === "string") {
          this.handleTextMessage(JSON.parse(event.data));
        } else if (event.data instanceof ArrayBuffer) {
          this.handleBinaryAudioChunk(event.data);
        }
      };
    });
  }

  async initMicrophone() {
    this.updateStatus("connecting", "Vui lòng cho phép quyền truy cập Microphone...");

    this.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
    try {
      this.audioContext = new AudioCtxClass({ sampleRate: 16000 });
      console.log("AudioContext created natively at 16000Hz");
    } catch (e) {
      console.warn("Native 16000Hz AudioContext not supported, using default:", e);
      this.audioContext = new AudioCtxClass();
    }

    if (this.audioContext.state === "suspended") {
      await this.audioContext.resume();
    }

    const source = this.audioContext.createMediaStreamSource(this.mediaStream);

    // Create Analyser for Visualizer
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = 256;
    source.connect(this.analyser);

    // DSP Filter Chain:
    // 1. High-Pass Filter at 85Hz: Cuts out mechanical hum, table vibrations, and HVAC rumble
    const highpass = this.audioContext.createBiquadFilter();
    highpass.type = "highpass";
    highpass.frequency.value = 85;
    source.connect(highpass);

    let processingInput = highpass;

    // 2. Anti-aliasing low-pass filter at 7.5kHz if sampleRate > 16kHz
    if (this.audioContext.sampleRate > 16000) {
      const lowpass = this.audioContext.createBiquadFilter();
      lowpass.type = "lowpass";
      lowpass.frequency.value = 7500;
      highpass.connect(lowpass);
      processingInput = lowpass;
    }

    // Audio processor node (buffer size 4096)
    this.scriptProcessor = this.audioContext.createScriptProcessor(4096, 1, 1);
    processingInput.connect(this.scriptProcessor);
    this.scriptProcessor.connect(this.audioContext.destination);

    this.userSpeechPcmChunks = [];

    this.scriptProcessor.onaudioprocess = (e) => {
      // Full-duplex: keep streaming audio even when bot is speaking so user can barge-in
      if (!this.isRecording) return;

      const inputData = e.inputBuffer.getChannelData(0);
      
      // If bot is actively playing audio through speakers, attenuate mic input
      // to eliminate acoustic echo feedback while preserving intentional user interruptions
      let audioForVad = inputData;
      if (this.isBotSpeaking) {
        audioForVad = new Float32Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          audioForVad[i] = inputData[i] * 0.35;
        }
      }

      const pcm16 = this.downsampleAndConvertToInt16(audioForVad, e.inputBuffer.sampleRate, 16000);

      // Buffer audio for User Audio Replay feature
      this.userSpeechPcmChunks.push(new Int16Array(pcm16));
      if (this.userSpeechPcmChunks.length > 50) {
        this.userSpeechPcmChunks.shift(); // Keep last ~12 seconds
      }

      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(pcm16.buffer);
      }
    };
  }

  encodePcm16ToWavBlob(chunks, sampleRate = 16000) {
    let totalLength = 0;
    for (const c of chunks) totalLength += c.length;

    const buffer = new ArrayBuffer(44 + totalLength * 2);
    const view = new DataView(buffer);

    // RIFF chunk descriptor
    this.writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + totalLength * 2, true);
    this.writeString(view, 8, "WAVE");

    // FMT sub-chunk
    this.writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM format
    view.setUint16(22, 1, true); // Mono channel
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // Byte rate
    view.setUint16(32, 2, true); // Block align
    view.setUint16(34, 16, true); // Bits per sample

    // Data sub-chunk
    this.writeString(view, 36, "data");
    view.setUint32(40, totalLength * 2, true);

    let offset = 44;
    for (const c of chunks) {
      for (let i = 0; i < c.length; i++) {
        view.setInt16(offset, c[i], true);
        offset += 2;
      }
    }

    return new Blob([view], { type: "audio/wav" });
  }

  writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  downsampleAndConvertToInt16(buffer, inputSampleRate, targetSampleRate) {
    if (inputSampleRate === targetSampleRate) {
      const pcm16 = new Int16Array(buffer.length);
      for (let i = 0; i < buffer.length; i++) {
        const s = Math.max(-1, Math.min(1, buffer[i]));
        pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      return pcm16;
    }

    // Linear interpolation resampling
    const ratio = inputSampleRate / targetSampleRate;
    const newLength = Math.round(buffer.length / ratio);
    const pcm16 = new Int16Array(newLength);

    for (let i = 0; i < newLength; i++) {
      const originIdx = i * ratio;
      const index = Math.floor(originIdx);
      const decimal = originIdx - index;
      const val = (buffer[index] || 0) + decimal * ((buffer[index + 1] || buffer[index]) - (buffer[index] || 0));
      const s = Math.max(-1, Math.min(1, val));
      pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }

    return pcm16;
  }

  handleTextMessage(data) {
    switch (data.type) {
      case "session_init":
        this.sessionId = data.session_id;
        this.sessionDisplay.innerText = `Phiên: ${this.sessionId.slice(0, 8)}...`;
        this.sessionIdInput.value = this.sessionId;
        if (data.voice && this.ttsVoiceSelect && !localStorage.getItem("voicebot_tts_voice")) {
          this.ttsVoiceSelect.value = data.voice;
          this.syncVoiceCards(data.voice);
        }
        if (data.rate && this.ttsRateSelect && !localStorage.getItem("voicebot_tts_rate")) {
          this.ttsRateSelect.value = data.rate;
          this.syncSpeedPills(data.rate);
        }
        break;

      case "status":
        this.updateStatus(data.status, data.detail);
        if (data.status === "thinking" || data.status === "processing") {
          this.showBotThinking();
        }
        break;

      case "bot_thinking":
        this.showBotThinking();
        break;

      case "bot_chunk":
        this.appendBotStreamChunk(data.chunk, data.text);
        break;

      case "vad_activity":
        if (data.is_speaking) {
          this.currentModeTag.innerText = "Đang nói...";
        }
        break;

      case "latency_metrics":
        this.updateLatencyMetrics(data);
        break;

      case "interrupt":
        console.log("Barge-in interrupt received from server:", data.reason);
        this.handleInterrupt(data.reason);
        break;

      case "user_transcript":
        // As soon as a new user question is recognized, STOP any previous answer audio immediately!
        this.stopCurrentAudio();
        let userAudioUrl = null;
        if (this.userSpeechPcmChunks.length > 0) {
          const wavBlob = this.encodePcm16ToWavBlob(this.userSpeechPcmChunks, 16000);
          userAudioUrl = URL.createObjectURL(wavBlob);
          this.createdObjectUrls.push(userAudioUrl);
          this.userSpeechPcmChunks = [];
        }
        this.appendMessage("user", data.text, userAudioUrl);
        this.showBotThinking();
        break;

      case "bot_response":
        this.finalizeBotResponse(data.text);
        break;

      case "tts_stream_start":
        // Stop any currently playing audio so new speech never overlaps!
        this.stopCurrentAudio();
        this.playbackQueue = [];
        this.currentSentenceChunks = [];
        this.allSentencesDone = false;
        this.isBotSpeaking = true;
        this.updateStatus("speaking", "Trợ lý đang phản hồi...");
        break;

      case "tts_sentence_start":
        this.currentSentenceChunks = [];
        break;

      case "tts_sentence_end":
        this.handleSentenceEnd(data.sentence_index);
        break;

      case "tts_stream_end":
      case "tts_all_done":
        this.allSentencesDone = true;
        if (!this.isPlayingQueue && this.playbackQueue.length === 0) {
          this.onPlaybackFinished();
        }
        break;
    }
  }

  updateLatencyMetrics(data) {
    if (this.sttMetricVal && data.stt_ms !== undefined) {
      this.sttMetricVal.innerText = `${data.stt_ms}ms`;
      this.sttMetricVal.className = "metric-num " + (data.stt_ms < 400 ? "highlight-text" : "");
    }
    if (this.llmMetricVal && data.llm_ms !== undefined) {
      this.llmMetricVal.innerText = `${data.llm_ms}ms`;
      this.llmMetricVal.className = "metric-num " + (data.llm_ms < 800 ? "highlight-text" : "");
    }
    if (this.ttsMetricVal && data.tts_ms !== undefined) {
      this.ttsMetricVal.innerText = `${data.tts_ms}ms`;
      this.ttsMetricVal.className = "metric-num " + (data.tts_ms < 400 ? "highlight-text" : "");
    }
    if (data.total_ms !== undefined) {
      const sec = (data.total_ms / 1000).toFixed(2);
      if (this.totalMetricVal) {
        this.totalMetricVal.innerText = `${sec}s`;
      }
      if (this.totalMetricDetailVal) {
        this.totalMetricDetailVal.innerText = `${data.total_ms}ms`;
      }
    }

    // Sync to Analytics Tab KPI cards
    const kpiStt = document.getElementById("kpiSttVal");
    const kpiLlm = document.getElementById("kpiLlmVal");
    const kpiTts = document.getElementById("kpiTtsVal");
    const kpiTotal = document.getElementById("kpiTotalVal");
    if (kpiStt && data.stt_ms !== undefined) kpiStt.innerText = `${data.stt_ms} ms`;
    if (kpiLlm && data.llm_ms !== undefined) kpiLlm.innerText = `${data.llm_ms} ms`;
    if (kpiTts && data.tts_ms !== undefined) kpiTts.innerText = `${data.tts_ms} ms`;
    if (kpiTotal && data.total_ms !== undefined) kpiTotal.innerText = `${data.total_ms} ms`;
  }

  handleSentenceEnd(sentenceIndex) {
    if (!this.isBotSpeaking || this.currentSentenceChunks.length === 0) return;

    const audioBlob = new Blob(this.currentSentenceChunks, { type: "audio/mpeg" });
    const audioUrl = URL.createObjectURL(audioBlob);
    this.createdObjectUrls.push(audioUrl);
    this.playbackQueue.push({ url: audioUrl, index: sentenceIndex });
    this.currentSentenceChunks = [];

    // If nothing is playing right now, play Sentence 1 immediately (<400ms)!
    if (!this.isPlayingQueue) {
      this.playNextInQueue();
    }
  }

  playNextInQueue() {
    if (!this.isBotSpeaking) return;

    if (this.playbackQueue.length === 0) {
      this.isPlayingQueue = false;
      if (this.allSentencesDone) {
        this.onPlaybackFinished();
      }
      return;
    }

    this.isPlayingQueue = true;
    const item = this.playbackQueue.shift();
    this.currentAudioUrl = item.url;
    this.currentAudioPlayer = new Audio(item.url);

    this.currentAudioPlayer.onended = () => {
      this.isPlayingQueue = false;
      this.playNextInQueue();
    };

    this.currentAudioPlayer.onerror = (e) => {
      console.warn("Queue item playback error:", e);
      this.isPlayingQueue = false;
      this.playNextInQueue();
    };

    this.currentAudioPlayer.play().catch((err) => {
      console.warn("Auto-play error on sentence item:", err);
      this.isPlayingQueue = false;
      this.playNextInQueue();
    });
  }

  stopCurrentAudio() {
    this.isPlayingQueue = false;
    this.playbackQueue = [];
    this.currentSentenceChunks = [];

    if (this.currentAudioPlayer) {
      try {
        this.currentAudioPlayer.onended = null;
        this.currentAudioPlayer.onerror = null;
        this.currentAudioPlayer.pause();
        this.currentAudioPlayer.currentTime = 0;
        this.currentAudioPlayer.src = "";
      } catch (e) {}
      this.currentAudioPlayer = null;
    }

    if (this.createdObjectUrls && this.createdObjectUrls.length > 0) {
      this.createdObjectUrls.forEach((url) => {
        try { URL.revokeObjectURL(url); } catch (e) {}
      });
      this.createdObjectUrls = [];
    }
    this.currentAudioUrl = null;
  }

  handleInterrupt(reason) {
    this.stopCurrentAudio();
    this.removeBotThinking();
    const streamBubble = document.getElementById("botStreamingBubble");
    if (streamBubble) {
      streamBubble.removeAttribute("id");
      streamBubble.classList.remove("streaming");
      const cursor = streamBubble.querySelector(".stream-cursor");
      if (cursor) cursor.remove();
    }
    this.allSentencesDone = true;
    this.isBotSpeaking = false;

    // Visual feedback for barge-in
    const card = document.querySelector(".assistant-hub-card");
    if (card) {
      card.classList.remove("speaking");
      card.classList.add("interrupted");
      setTimeout(() => card.classList.remove("interrupted"), 1200);
    }

    this.updateStatus("listening", "Đã ngắt lời! Đang lắng nghe câu tiếp theo...");
  }

  handleBinaryAudioChunk(chunkArrayBuffer) {
    if (!this.isBotSpeaking) return;
    this.currentSentenceChunks.push(chunkArrayBuffer);
  }

  onPlaybackFinished() {
    if (this.playbackQueue.length > 0 || this.isPlayingQueue) {
      return;
    }
    this.isBotSpeaking = false;
    this.currentSentenceChunks = [];
    this.playbackQueue = [];
    this.isPlayingQueue = false;
    this.stopCurrentAudio();

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "playback_ended" }));
    }
    this.updateStatus("listening", "Đang lắng nghe... Hãy nói tiếp nhé!");
  }

  updateStatus(status, detail) {
    const card = document.querySelector(".assistant-hub-card");
    if (card) card.classList.remove("speaking");

    switch (status) {
      case "listening":
        this.currentModeTag.innerText = "Lắng nghe";
        this.indicatorIcon.innerText = "🎙️";
        this.indicatorText.innerText = detail || "Đang lắng nghe giọng nói của bạn...";
        break;

      case "processing":
      case "thinking":
        this.currentModeTag.innerText = "Đang xử lý";
        this.indicatorIcon.innerText = "💭";
        this.indicatorText.innerText = detail || `${this.getAssistantName()} đang suy nghĩ câu trả lời...`;
        break;

      case "speaking":
        this.currentModeTag.innerText = `${this.getAssistantName()} đang nói`;
        if (card) card.classList.add("speaking");
        this.indicatorIcon.innerText = "🔊";
        this.indicatorText.innerText = detail || `${this.getAssistantName()} đang phản hồi...`;
        break;

      case "disconnected":
        this.currentModeTag.innerText = "Sẵn sàng";
        this.indicatorIcon.innerText = "✨";
        this.indicatorText.innerText = detail || "Nhấn nút microphone bên dưới để bắt đầu";
        break;
    }
  }

  showBotThinking() {
    this.removeBotThinking();

    // Remove welcome card if it's the first message
    const welcomeCard = this.chatContainer.querySelector(".system-welcome-card");
    if (welcomeCard) welcomeCard.remove();

    const bubble = document.createElement("div");
    bubble.className = "message-bubble bot-bubble thinking-bubble";
    bubble.id = "botThinkingBubble";

    const meta = document.createElement("div");
    meta.className = "message-meta";
    const sender = `🤖 ${this.getAssistantName()}`;
    const time = new Date().toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" });
    meta.innerHTML = `<span>${sender}</span><span>•</span><span>${time}</span>`;
    bubble.appendChild(meta);

    const content = document.createElement("div");
    content.className = "message-content";
    content.innerHTML = `
      <span>Trợ lý AI đang suy nghĩ...</span>
      <div class="thinking-spinner">
        <span></span><span></span><span></span>
      </div>
    `;
    bubble.appendChild(content);

    this.chatContainer.appendChild(bubble);
    this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
  }

  removeBotThinking() {
    const existing = document.getElementById("botThinkingBubble");
    if (existing) existing.remove();
  }

  appendBotStreamChunk(chunk, fullAccumulated = null) {
    this.removeBotThinking();

    let streamBubble = document.getElementById("botStreamingBubble");
    if (!streamBubble) {
      const welcomeCard = this.chatContainer.querySelector(".system-welcome-card");
      if (welcomeCard) welcomeCard.remove();

      streamBubble = document.createElement("div");
      streamBubble.className = "message-bubble bot-bubble streaming";
      streamBubble.id = "botStreamingBubble";

      const meta = document.createElement("div");
      meta.className = "message-meta";
      const sender = `🤖 ${this.getAssistantName()}`;
      const time = new Date().toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" });
      meta.innerHTML = `<span>${sender}</span><span>•</span><span>${time}</span>`;
      streamBubble.appendChild(meta);

      const content = document.createElement("div");
      content.className = "message-content";
      content.innerHTML = `<span class="stream-text"></span><span class="stream-cursor"></span>`;
      streamBubble.appendChild(content);

      this.chatContainer.appendChild(streamBubble);
    }

    const textSpan = streamBubble.querySelector(".stream-text");
    if (textSpan) {
      if (fullAccumulated !== null && fullAccumulated !== undefined) {
        textSpan.innerText = fullAccumulated;
      } else {
        textSpan.innerText += chunk;
      }
    }

    this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
  }

  finalizeBotResponse(fullText) {
    this.removeBotThinking();

    const streamBubble = document.getElementById("botStreamingBubble");
    if (streamBubble) {
      streamBubble.removeAttribute("id");
      streamBubble.classList.remove("streaming");
      const cursor = streamBubble.querySelector(".stream-cursor");
      if (cursor) cursor.remove();
      const textSpan = streamBubble.querySelector(".stream-text");
      if (textSpan) {
        textSpan.innerText = fullText;
      }

      // Save to chat history
      const sender = `🤖 ${this.getAssistantName()}`;
      this.chatHistory.push({
        id: Date.now(),
        role: "bot",
        senderName: sender,
        text: fullText,
        timestamp: new Date().toLocaleTimeString("vi-VN"),
      });
      try {
        localStorage.setItem("voicebot_chat_history", JSON.stringify(this.chatHistory));
      } catch (e) {}

      const countBadge = document.getElementById("historyCountBadge");
      if (countBadge) countBadge.innerText = this.chatHistory.length;
    } else {
      this.appendMessage("bot", fullText);
    }

    this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
  }

  appendMessage(role, text, audioUrl = null, saveHistory = true, customSenderName = null) {
    if (!text || !text.trim()) return;

    // Remove welcome card if it's the first message
    const welcomeCard = this.chatContainer.querySelector(".system-welcome-card");
    if (welcomeCard) {
      welcomeCard.remove();
    }

    const bubble = document.createElement("div");
    bubble.className = `message-bubble ${role === "user" ? "user-bubble" : "bot-bubble"}`;

    // Message meta info
    const meta = document.createElement("div");
    meta.className = "message-meta";
    const sender = role === "user" ? "👤 Bạn" : (customSenderName || `🤖 ${this.getAssistantName()}`);
    const time = new Date().toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" });
    meta.innerHTML = `<span>${sender}</span><span>•</span><span>${time}</span>`;
    bubble.appendChild(meta);

    // Message content body
    const content = document.createElement("div");
    content.className = "message-content";
    content.innerText = text;
    bubble.appendChild(content);

    // Audio Replay Button for user speech
    if (role === "user" && audioUrl) {
      const replayBtn = document.createElement("button");
      replayBtn.className = "audio-replay-btn";
      replayBtn.innerHTML = "▶️ Nghe lại";
      replayBtn.title = "Nghe lại đoạn ghi âm vừa nói của bạn";
      replayBtn.addEventListener("click", () => {
        const audio = new Audio(audioUrl);
        audio.play().catch((err) => console.warn("Replay error:", err));
      });
      bubble.appendChild(replayBtn);
    }

    this.chatContainer.appendChild(bubble);
    this.chatContainer.scrollTop = this.chatContainer.scrollHeight;

    if (saveHistory) {
      this.chatHistory.push({
        id: Date.now(),
        role: role,
        senderName: sender,
        text: text,
        timestamp: new Date().toLocaleTimeString("vi-VN"),
      });
      try {
        localStorage.setItem("voicebot_chat_history", JSON.stringify(this.chatHistory));
      } catch (e) {}

      const countBadge = document.getElementById("historyCountBadge");
      if (countBadge) countBadge.innerText = this.chatHistory.length;
    }
  }

  clearChat() {
    if (confirm("Bạn có chắc chắn muốn làm mới toàn bộ lịch sử trò chuyện không?")) {
      this.chatHistory = [];
      try {
        localStorage.removeItem("voicebot_chat_history");
      } catch (e) {}

      const countBadge = document.getElementById("historyCountBadge");
      if (countBadge) countBadge.innerText = "0";

      const historyContainer = document.getElementById("historySessionList");
      if (historyContainer) this.renderHistoryList();
      this.chatContainer.innerHTML = `
        <div class="system-welcome-card">
          <div class="welcome-icon">💬</div>
          <h3>Lịch sử đã được làm mới</h3>
          <p>Hãy bấm nút microphone hoặc nhập tin nhắn để bắt đầu cuộc trò chuyện mới cùng trợ lý nhé!</p>
        </div>
      `;
      this.showToast("Đã làm mới lịch sử trò chuyện.", "info");
    }
  }

  exportAsText() {
    if (this.chatHistory.length === 0) {
      this.showToast("Chưa có tin nhắn nào để xuất.", "warning");
      return;
    }

    const dateStr = new Date().toLocaleString("vi-VN");
    let content = "=========================================================\n";
    content += "TRỢ LÝ ẢO TIẾNG VIỆT - LỊCH SỬ HỘI THOẠI\n";
    content += `Thời gian xuất: ${dateStr}\n`;
    content += `Phiên: ${this.sessionId || "N/A"}\n`;
    content += "=========================================================\n\n";

    for (const msg of this.chatHistory) {
      let sender = "BẠN";
      if (msg.role !== "user") {
        sender = msg.senderName ? msg.senderName.replace("🤖 ", "").toUpperCase() : this.getAssistantName().toUpperCase();
      }
      content += `[${msg.timestamp}] ${sender}:\n${msg.text}\n\n`;
    }

    this.triggerDownload(content, `tro_ly_hoi_thoai_${Date.now()}.txt`, "text/plain;charset=utf-8");
    this.showToast("Đã xuất file văn bản TXT thành công!", "success");
  }

  exportAsJson() {
    if (this.chatHistory.length === 0) {
      this.showToast("Chưa có dữ liệu để xuất.", "warning");
      return;
    }

    const exportData = {
      exported_at: new Date().toISOString(),
      session_id: this.sessionId,
      total_messages: this.chatHistory.length,
      conversation: this.chatHistory,
    };

    const jsonStr = JSON.stringify(exportData, null, 2);
    this.triggerDownload(jsonStr, `tro_ly_hoi_thoai_${Date.now()}.json`, "application/json;charset=utf-8");
    this.showToast("Đã xuất file dữ liệu JSON thành công!", "success");
  }

  triggerDownload(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  // Fluid Organic Soundwave Visualizer
  initVisualizer() {
    const draw = () => {
      requestAnimationFrame(draw);

      if (!this.canvas || !this.canvasCtx) return;

      const width = this.canvas.width;
      const height = this.canvas.height;
      this.canvasCtx.clearRect(0, 0, width, height);

      this.visualizerTime += 0.04;

      let bufferLength = 64;
      let dataArray = new Uint8Array(bufferLength);
      let isLiveAudio = false;

      if (this.analyser && this.isRecording && !this.isBotSpeaking) {
        this.analyser.getByteFrequencyData(dataArray);
        isLiveAudio = true;
      }

      // Draw smooth harmonic sine waves
      const centerY = height / 2;
      const waveCount = this.isBotSpeaking ? 3 : isLiveAudio ? 2 : 1;

      for (let w = 0; w < waveCount; w++) {
        this.canvasCtx.beginPath();
        this.canvasCtx.lineWidth = 2.5;

        // Gradient color for waves
        const gradient = this.canvasCtx.createLinearGradient(0, 0, width, 0);
        if (this.isBotSpeaking) {
          gradient.addColorStop(0, "rgba(16, 185, 129, 0.2)");
          gradient.addColorStop(0.5, "rgba(6, 182, 212, 0.85)");
          gradient.addColorStop(1, "rgba(99, 102, 241, 0.2)");
        } else if (isLiveAudio) {
          gradient.addColorStop(0, "rgba(99, 102, 241, 0.2)");
          gradient.addColorStop(0.5, "rgba(56, 189, 248, 0.85)");
          gradient.addColorStop(1, "rgba(16, 185, 129, 0.2)");
        } else {
          gradient.addColorStop(0, "rgba(255, 255, 255, 0.05)");
          gradient.addColorStop(0.5, "rgba(99, 102, 241, 0.35)");
          gradient.addColorStop(1, "rgba(255, 255, 255, 0.05)");
        }

        this.canvasCtx.strokeStyle = gradient;

        for (let x = 0; x < width; x += 3) {
          const progress = x / width;
          // Sine wave envelope (tapers smoothly at edges)
          const envelope = Math.sin(progress * Math.PI);

          let amplitude = 6;
          if (this.isBotSpeaking) {
            amplitude = 18 + Math.sin(this.visualizerTime * 2 + w) * 8;
          } else if (isLiveAudio) {
            const freqIdx = Math.floor(progress * bufferLength);
            const freqVal = (dataArray[freqIdx] || 0) / 255;
            amplitude = 4 + freqVal * 32;
          }

          const frequency = 0.025 + w * 0.01;
          const y = centerY + Math.sin(x * frequency + this.visualizerTime * (1.5 + w * 0.5)) * amplitude * envelope;

          if (x === 0) {
            this.canvasCtx.moveTo(x, y);
          } else {
            this.canvasCtx.lineTo(x, y);
          }
        }

        this.canvasCtx.stroke();
      }
    };

    draw();
  }

  /* ==============================================================================
     SAAS STUDIO EXTENSIONS: TABS, AUTH & SUBPAGE ACTIONS
     ============================================================================== */

  initTabNavigation() {
    const navItems = document.querySelectorAll(".sidebar-nav .nav-item");
    const panes = document.querySelectorAll(".tab-pane");
    const topbarTitle = document.getElementById("topbarTabTitle");

    const tabTitles = {
      voicebot: "🎙️ Đàm thoại trực tiếp",
      history: "💬 Lịch sử các phiên đối thoại",
      analytics: "📊 Phân tích Âm học & Đo lường Độ trễ",
      settings: "⚙️ Cài đặt Giọng nói & Hệ thống",
      docs: "📖 Kiến trúc Hệ thống & Tài liệu Kỹ thuật",
    };

    navItems.forEach((btn) => {
      btn.addEventListener("click", () => {
        const targetTab = btn.dataset.tab;
        if (!targetTab) return;

        // Active state on nav buttons
        navItems.forEach((item) => item.classList.remove("active"));
        btn.classList.add("active");

        // Active state on panes
        panes.forEach((pane) => pane.classList.remove("active"));
        const targetPane = document.getElementById(`pane${targetTab.charAt(0).toUpperCase() + targetTab.slice(1)}`);
        if (targetPane) {
          targetPane.classList.add("active");
        }

        // Update Topbar Title
        if (topbarTitle && tabTitles[targetTab]) {
          topbarTitle.innerText = tabTitles[targetTab];
        }

        // Tab-specific handlers
        if (targetTab === "history") {
          this.renderHistoryList();
        } else if (targetTab === "voicebot") {
          this.resizeCanvas();
        }

        // Mobile sidebar close
        const sidebar = document.getElementById("appSidebar");
        if (sidebar && window.innerWidth <= 1024) {
          sidebar.classList.remove("open");
        }
      });
    });

    // Mobile sidebar toggle
    const toggleBtn = document.getElementById("sidebarToggleBtn");
    const sidebar = document.getElementById("appSidebar");
    if (toggleBtn && sidebar) {
      toggleBtn.addEventListener("click", () => {
        sidebar.classList.toggle("open");
      });
    }
  }

  initAuth() {
    this.token = localStorage.getItem("voicebot_token") || null;
    const savedUser = localStorage.getItem("voicebot_user");
    this.currentUser = savedUser ? JSON.parse(savedUser) : {
      username: "admin",
      name: "Nguyễn Văn Đạt",
      role: "Kỹ sư AI / Admin",
      avatar: "👨‍💻",
    };

    // If no token exists on first load, initialize as logged in with admin demo
    if (!this.token) {
      this.token = "token_demo_admin";
      localStorage.setItem("voicebot_token", this.token);
      localStorage.setItem("voicebot_user", JSON.stringify(this.currentUser));
    }

    this.renderUserWidget();

    // DOM Elements for Login Modal
    const loginModal = document.getElementById("loginModal");
    const closeLoginBtn = document.getElementById("closeLoginModalBtn");
    const authForm = document.getElementById("authLoginForm");
    const usernameInput = document.getElementById("loginUsernameInput");
    const passwordInput = document.getElementById("loginPasswordInput");
    const toggleEyeBtn = document.getElementById("togglePasswordVisibilityBtn");
    const quickDemoBtn = document.getElementById("quickDemoLoginBtn");
    const logoutBtn = document.getElementById("sidebarLogoutBtn");
    const loginTriggerBtn = document.getElementById("sidebarLoginTriggerBtn");
    const topbarUserChip = document.getElementById("topbarUserChip");

    if (closeLoginBtn && loginModal) {
      closeLoginBtn.addEventListener("click", () => loginModal.classList.add("hidden"));
    }

    if (loginTriggerBtn && loginModal) {
      loginTriggerBtn.addEventListener("click", () => loginModal.classList.remove("hidden"));
    }

    if (topbarUserChip && loginModal) {
      topbarUserChip.addEventListener("click", () => {
        if (!this.token) {
          loginModal.classList.remove("hidden");
        } else {
          this.showToast(`Tài khoản: ${this.currentUser.name} (${this.currentUser.role})`, "info");
        }
      });
    }

    if (toggleEyeBtn && passwordInput) {
      toggleEyeBtn.addEventListener("click", () => {
        const isPass = passwordInput.type === "password";
        passwordInput.type = isPass ? "text" : "password";
        toggleEyeBtn.innerText = isPass ? "🙈" : "👁️";
      });
    }

    // Submit Login
    const handleLogin = async (username, password) => {
      try {
        const res = await fetch("/api/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username, password }),
        });

        if (res.ok) {
          const data = await res.json();
          this.token = data.token;
          this.currentUser = data.user;
          localStorage.setItem("voicebot_token", this.token);
          localStorage.setItem("voicebot_user", JSON.stringify(this.currentUser));
          this.renderUserWidget();
          if (loginModal) loginModal.classList.add("hidden");
          this.showToast(data.message || `Đăng nhập thành công! Chào ${this.currentUser.name}`, "success");
        } else {
          const errData = await res.json().catch(() => ({}));
          this.showToast(errData.detail || "Đăng nhập thất bại. Kiểm tra lại thông tin!", "warning");
        }
      } catch (err) {
        // Fallback local login for standalone usage
        this.token = "token_offline";
        this.currentUser = {
          username: username,
          name: username.charAt(0).toUpperCase() + username.slice(1),
          role: "Thành viên Voicebot",
          avatar: "🧑‍🚀",
        };
        localStorage.setItem("voicebot_token", this.token);
        localStorage.setItem("voicebot_user", JSON.stringify(this.currentUser));
        this.renderUserWidget();
        if (loginModal) loginModal.classList.add("hidden");
        this.showToast(`Đăng nhập thành công: ${this.currentUser.name}`, "success");
      }
    };

    if (authForm) {
      authForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const u = usernameInput ? usernameInput.value.trim() : "";
        const p = passwordInput ? passwordInput.value.trim() : "";
        if (!u || !p) {
          this.showToast("Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu", "warning");
          return;
        }
        handleLogin(u, p);
      });
    }

    if (quickDemoBtn) {
      quickDemoBtn.addEventListener("click", () => {
        if (usernameInput) usernameInput.value = "admin";
        if (passwordInput) passwordInput.value = "123";
        handleLogin("admin", "123");
      });
    }

    // Logout Action
    if (logoutBtn) {
      logoutBtn.addEventListener("click", async () => {
        try {
          if (this.token) {
            await fetch("/api/auth/logout", {
              method: "POST",
              headers: { Authorization: `Bearer ${this.token}` },
            }).catch(() => {});
          }
        } catch (_) {}

        this.token = null;
        localStorage.removeItem("voicebot_token");
        localStorage.removeItem("voicebot_user");

        if (this.isRecording) {
          this.stopConversation();
        }

        this.renderUserWidget();
        if (loginModal) loginModal.classList.remove("hidden");
        this.showToast("Đã đăng xuất tài khoản an toàn.", "info");
      });
    }
  }

  renderUserWidget() {
    const profileBox = document.getElementById("sidebarUserProfile");
    const promptBox = document.getElementById("sidebarLoginPrompt");
    const sidebarAvatar = document.getElementById("sidebarUserAvatar");
    const sidebarName = document.getElementById("sidebarUserName");
    const sidebarRole = document.getElementById("sidebarUserRole");
    const topbarAvatar = document.getElementById("topbarUserAvatar");
    const topbarName = document.getElementById("topbarUserName");

    if (this.token && this.currentUser) {
      if (profileBox) profileBox.classList.remove("hidden");
      if (promptBox) promptBox.classList.add("hidden");

      if (sidebarAvatar) sidebarAvatar.innerText = this.currentUser.avatar || "👨‍💻";
      if (sidebarName) sidebarName.innerText = this.currentUser.name || "Nguyễn Văn Đạt";
      if (sidebarRole) sidebarRole.innerText = this.currentUser.role || "Kỹ sư AI / Admin";

      if (topbarAvatar) topbarAvatar.innerText = this.currentUser.avatar || "👨‍💻";
      if (topbarName) topbarName.innerText = this.currentUser.name || "Nguyễn Văn Đạt";
    } else {
      if (profileBox) profileBox.classList.add("hidden");
      if (promptBox) promptBox.classList.remove("hidden");

      if (topbarAvatar) topbarAvatar.innerText = "🔒";
      if (topbarName) topbarName.innerText = "Chưa đăng nhập";
    }
  }

  renderHistoryList() {
    const container = document.getElementById("historySessionList");
    const countBadge = document.getElementById("historyCountBadge");
    if (!container) return;

    if (countBadge) {
      countBadge.innerText = this.chatHistory.length;
    }

    if (!this.chatHistory || this.chatHistory.length === 0) {
      container.innerHTML = `
        <div style="text-align: center; padding: 40px 20px; color: var(--text-tertiary);">
          <div style="font-size: 2.2rem; margin-bottom: 8px;">💬</div>
          <p style="font-size: 0.9rem;">Chưa có tin nhắn nào trong phiên đối thoại hiện tại.</p>
          <small>Bấm vào tab 🎙️ Đàm thoại để bắt đầu nói chuyện với trợ lý!</small>
        </div>
      `;
      return;
    }

    const searchInput = document.getElementById("historySearchInput");
    const query = searchInput ? searchInput.value.trim().toLowerCase() : "";

    const filtered = this.chatHistory.filter((item) => {
      if (!query) return true;
      return (item.text && item.text.toLowerCase().includes(query)) ||
             (item.senderName && item.senderName.toLowerCase().includes(query));
    });

    if (filtered.length === 0) {
      container.innerHTML = `<p style="text-align: center; color: var(--text-tertiary); padding: 20px;">Không tìm thấy tin nhắn khớp với từ khóa "${query}".</p>`;
      return;
    }

    container.innerHTML = filtered.map((item, idx) => `
      <div class="history-session-card">
        <div class="session-info-left">
          <div class="session-title">
            <span>${item.role === "user" ? "👤 Bạn" : "🤖 Trợ lý AI"}</span>: "${item.text.length > 80 ? item.text.substring(0, 80) + '...' : item.text}"
          </div>
          <div class="session-meta-row">
            <span>⏱️ ${item.timestamp || 'Vừa xong'}</span>
            <span>•</span>
            <span>Vai trò: ${item.role}</span>
          </div>
        </div>
        <div>
          <button class="btn-tool-pill" onclick="window.voicebot.jumpToChat('${item.id || idx}')">Xem</button>
        </div>
      </div>
    `).join("");
  }

  saveSettings() {
    if (this.wsUrlInput) {
      this.wsUrl = this.wsUrlInput.value.trim();
      localStorage.setItem("voicebot_ws_url", this.wsUrl);
    }
    const silenceVal = this.vadSensitivityInput ? parseInt(this.vadSensitivityInput.value) : 600;
    const thresholdVal = this.vadThresholdInput ? parseFloat(this.vadThresholdInput.value) : 0.45;
    const voiceVal = this.ttsVoiceSelect ? this.ttsVoiceSelect.value : "vi-VN-HoaiMyNeural";
    const rateVal = this.ttsRateSelect ? this.ttsRateSelect.value : "+15%";

    localStorage.setItem("voicebot_vad_silence", silenceVal);
    localStorage.setItem("voicebot_vad_threshold", thresholdVal);
    localStorage.setItem("voicebot_tts_voice", voiceVal);
    localStorage.setItem("voicebot_tts_rate", rateVal);

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      if (this.sessionIdInput && this.sessionIdInput.value.trim()) {
        this.ws.send(JSON.stringify({
          type: "set_session",
          session_id: this.sessionIdInput.value.trim()
        }));
      }
      this.sendSettingsToServer();
    }
    if (this.settingsModal) {
      this.settingsModal.classList.add("hidden");
    }
    this.showToast("Đã lưu và áp dụng thiết lập thành công!", "success");
  }

  jumpToChat(id) {
    const voicebotNavBtn = document.querySelector('.sidebar-nav .nav-item[data-tab="voicebot"]');
    if (voicebotNavBtn) {
      voicebotNavBtn.click();
    }
  }

  initSubpageActions() {
    // Search input filter in History tab
    const searchInput = document.getElementById("historySearchInput");
    if (searchInput) {
      searchInput.addEventListener("input", () => this.renderHistoryList());
    }

    // Refresh history
    const refreshBtn = document.getElementById("historyRefreshBtn");
    if (refreshBtn) {
      refreshBtn.addEventListener("click", () => {
        this.renderHistoryList();
        this.showToast("Đã làm mới danh sách hội thoại!", "info");
      });
    }

    // Clear all history
    const clearAllBtn = document.getElementById("historyClearAllBtn");
    if (clearAllBtn) {
      clearAllBtn.addEventListener("click", () => this.clearChat());
    }

    // Reset analytics
    const resetAnalyticsBtn = document.getElementById("analyticsResetBtn");
    if (resetAnalyticsBtn) {
      resetAnalyticsBtn.addEventListener("click", () => {
        const kpiStt = document.getElementById("kpiSttVal");
        const kpiLlm = document.getElementById("kpiLlmVal");
        const kpiTts = document.getElementById("kpiTtsVal");
        const kpiTotal = document.getElementById("kpiTotalVal");
        if (kpiStt) kpiStt.innerText = "-- ms";
        if (kpiLlm) kpiLlm.innerText = "-- ms";
        if (kpiTts) kpiTts.innerText = "-- ms";
        if (kpiTotal) kpiTotal.innerText = "-- ms";
        this.showToast("Đã làm mới bộ đo lường âm học!", "info");
      });
    }
  }
}

// Initialize on DOM load
window.addEventListener("DOMContentLoaded", () => {
  window.voicebot = new VoicebotClient();
});

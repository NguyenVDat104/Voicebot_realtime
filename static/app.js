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
  }

  getAssistantName() {
    const voice = this.ttsVoiceSelect ? this.ttsVoiceSelect.value : (localStorage.getItem("voicebot_tts_voice") || "vi-VN-HoaiMyNeural");
    if (voice === "vi-VN-NamMinhNeural") {
      return "Nguyễn Đạt";
    }
    return "Ngọc Trinh";
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

    // Settings Modal open/close/save
    this.toggleSettingsBtn.addEventListener("click", () => {
      this.settingsModal.classList.remove("hidden");
    });
    this.closeSettingsBtn.addEventListener("click", () => {
      this.settingsModal.classList.add("hidden");
    });
    this.settingsModal.addEventListener("click", (e) => {
      if (e.target === this.settingsModal) {
        this.settingsModal.classList.add("hidden");
      }
    });

    this.saveSettingsBtn.addEventListener("click", () => {
      this.wsUrl = this.wsUrlInput.value.trim();
      localStorage.setItem("voicebot_ws_url", this.wsUrl);
      const silenceVal = this.vadSensitivityInput ? parseInt(this.vadSensitivityInput.value) : 600;
      const thresholdVal = this.vadThresholdInput ? parseFloat(this.vadThresholdInput.value) : 0.45;
      const voiceVal = this.ttsVoiceSelect ? this.ttsVoiceSelect.value : "vi-VN-HoaiMyNeural";
      const rateVal = this.ttsRateSelect ? this.ttsRateSelect.value : "+15%";

      localStorage.setItem("voicebot_vad_silence", silenceVal);
      localStorage.setItem("voicebot_vad_threshold", thresholdVal);
      localStorage.setItem("voicebot_tts_voice", voiceVal);
      localStorage.setItem("voicebot_tts_rate", rateVal);

      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        if (this.sessionIdInput.value.trim()) {
          this.ws.send(JSON.stringify({
            type: "set_session",
            session_id: this.sessionIdInput.value.trim()
          }));
        }
        this.sendSettingsToServer();
      }
      this.settingsModal.classList.add("hidden");
      this.showToast("Đã lưu và áp dụng thiết lập thành công!", "success");
    });

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
      const pcm16 = this.downsampleAndConvertToInt16(inputData, e.inputBuffer.sampleRate, 16000);

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
        break;

      case "bot_response":
        this.appendMessage("bot", data.text);
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
      this.sttMetricVal.className = "metric-value " + (data.stt_ms < 400 ? "metric-fast" : data.stt_ms < 1000 ? "metric-normal" : "metric-slow");
    }
    if (this.llmMetricVal && data.llm_ms !== undefined) {
      this.llmMetricVal.innerText = `${data.llm_ms}ms`;
      this.llmMetricVal.className = "metric-value " + (data.llm_ms < 800 ? "metric-fast" : data.llm_ms < 2000 ? "metric-normal" : "metric-slow");
    }
    if (this.ttsMetricVal && data.tts_ms !== undefined) {
      this.ttsMetricVal.innerText = `${data.tts_ms}ms`;
      this.ttsMetricVal.className = "metric-value " + (data.tts_ms < 400 ? "metric-fast" : data.tts_ms < 800 ? "metric-normal" : "metric-slow");
    }
    if (data.total_ms !== undefined) {
      const sec = (data.total_ms / 1000).toFixed(2);
      if (this.totalMetricVal) {
        this.totalMetricVal.innerText = `${sec}s`;
      }
      if (this.totalMetricDetailVal) {
        this.totalMetricDetailVal.innerText = `${data.total_ms}ms`;
        this.totalMetricDetailVal.className = "metric-value " + (data.total_ms < 1500 ? "metric-fast" : data.total_ms < 3000 ? "metric-normal" : "metric-slow");
      }
    }
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
    }
  }

  clearChat() {
    if (confirm("Bạn có chắc chắn muốn làm mới toàn bộ lịch sử trò chuyện không?")) {
      this.chatHistory = [];
      try {
        localStorage.removeItem("voicebot_chat_history");
      } catch (e) {}
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
}

// Initialize on DOM load
window.addEventListener("DOMContentLoaded", () => {
  window.voicebot = new VoicebotClient();
});

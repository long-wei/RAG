class ChatApp {
    constructor() {
        this.currentQuery = '';
        this.currentAnswer = '';
        this.currentChunks = [];
        this.currentRating = 0;
        this.recognition = null;
        this.synth = window.speechSynthesis;
        this.isSpeaking = false;
        this.init();
    }

    init() {
        // 绑定事件
        document.getElementById('sendButton').addEventListener('click', () => this.sendMessage());
        document.getElementById('userInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });
        
        // 语音输入按钮事件
        document.getElementById('voiceInputButton').addEventListener('click', () => this.startVoiceInput());
        
        // 停止播放按钮事件
        document.getElementById('stopTTSButton').addEventListener('click', () => this.stopSpeech());
        
        // 星级评分事件
        const stars = document.querySelectorAll('.star');
        stars.forEach(star => {
            star.addEventListener('click', () => {
                const rating = parseInt(star.getAttribute('data-rating'));
                this.currentRating = rating;
                this.updateStars(rating);
            });
            
            star.addEventListener('mouseover', () => {
                const rating = parseInt(star.getAttribute('data-rating'));
                this.highlightStars(rating);
            });
        });
        
        // 鼠标离开时恢复选中状态
        document.querySelector('.rating-stars').addEventListener('mouseleave', () => {
            this.updateStars(this.currentRating);
        });
        
        // 提交评分
        document.getElementById('submitRating').addEventListener('click', () => this.submitRating());
        
        // 初始化语音识别
        this.initSpeechRecognition();
    }

    initSpeechRecognition() {
        // 检查浏览器是否支持语音识别
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            this.recognition = new SpeechRecognition();
            this.recognition.continuous = false;
            this.recognition.interimResults = false;
            this.recognition.lang = 'zh-CN'; // 设置为中文识别
            
            this.recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                document.getElementById('userInput').value = transcript;
                this.appendMessage(`[语音输入] ${transcript}`, 'user');
                this.sendMessage();
            };
            
            this.recognition.onerror = (event) => {
                console.error('语音识别出错:', event.error);
                let errorMsg = event.error;
                if (event.error === 'not-allowed') {
                    errorMsg = '麦克风访问被拒绝，请检查浏览器权限设置并确保网站通过HTTPS访问';
                } else if (event.error === 'permission-denied') {
                    errorMsg = '麦克风权限被拒绝，请在浏览器设置中允许麦克风访问';
                } else if (event.error === 'no-speech') {
                    errorMsg = '未检测到语音，请重试';
                }
                this.appendMessage('语音识别出错: ' + errorMsg, 'system');
            };
            
            this.recognition.onend = () => {
                document.getElementById('voiceInputButton').innerHTML = '<i class="fas fa-microphone"></i> 语音输入';
            };
        } else {
            console.warn('浏览器不支持语音识别功能');
            this.appendMessage('当前浏览器不支持语音识别功能，请使用最新版Chrome浏览器并通过HTTPS访问', 'system');
            document.getElementById('voiceInputButton').disabled = true;
        }
    }

    startVoiceInput() {
        if (this.recognition) {
            this.recognition.start();
            document.getElementById('voiceInputButton').innerHTML = '<i class="fas fa-microphone"></i> 识别中...';
        } else {
            this.appendMessage('浏览器不支持语音识别功能', 'system');
        }
    }

    stopSpeech() {
        if (this.synth.speaking) {
            this.synth.cancel();
            this.isSpeaking = false;
        }
    }

    async sendMessage() {
        const userInput = document.getElementById('userInput').value.trim();
        if (!userInput) return;

        // 显示用户消息
        // 注意：语音输入的消息已经在startVoiceInput中显示过了
        if (!userInput.startsWith('[语音输入]')) {
            this.appendMessage(userInput, 'user');
        }
        document.getElementById('userInput').value = '';

        // 显示打字指示器
        document.getElementById('typingIndicator').style.display = 'block';

        // 隐藏评分区域
        document.getElementById('ratingSection').style.display = 'none';

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: userInput.replace('[语音输入] ', '') })
            });

            const data = await response.json();
            document.getElementById('typingIndicator').style.display = 'none';

            if (!response.ok) {
                this.appendMessage('系统错误: ' + (data.error || '未知错误'), 'system');
                return;
            }

            if (data.error) {
                this.appendMessage('系统错误: ' + data.error, 'system');
            } else {
                this.appendMessage(data.answer, 'bot');
                this.showSources(data.chunks);

                // 保存当前对话信息
                this.currentQuery = data.query;
                this.currentAnswer = data.answer;
                this.currentChunks = data.chunks;

                // 显示评分区域
                document.getElementById('ratingSection').style.display = 'block';
                this.currentRating = 0;
                this.updateStars(0);
                
                // 自动播放回答的语音
                this.speakText(data.answer);
            }
        } catch (error) {
            document.getElementById('typingIndicator').style.display = 'none';
            this.appendMessage('网络错误: ' + error.message, 'system');
        }
    }

    speakText(text) {
        if ('speechSynthesis' in window) {
            // 停止当前正在播放的语音
            if (this.synth.speaking) {
                this.synth.cancel();
            }
            
            const utterThis = new SpeechSynthesisUtterance(text);
            utterThis.lang = 'zh-CN';
            utterThis.rate = 1;
            utterThis.pitch = 1;
            
            utterThis.onend = () => {
                this.isSpeaking = false;
            };
            
            utterThis.onerror = (event) => {
                console.error('语音播放出错:', event);
                this.isSpeaking = false;
            };
            
            this.synth.speak(utterThis);
            this.isSpeaking = true;
        } else {
            console.warn('浏览器不支持语音合成功能');
        }
    }

    appendMessage(message, sender) {
        const chatMessages = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');

        if (sender === 'user') {
            messageDiv.classList.add('user-message');
            messageDiv.innerHTML = `<strong>您:</strong><br>${this.escapeHtml(message)}`;
        } else if (sender === 'bot') {
            messageDiv.classList.add('bot-message');
            messageDiv.innerHTML = `<strong>助手:</strong><br>${this.formatMessage(message)}`;
        } else {
            messageDiv.classList.add('system-message');
            messageDiv.textContent = message;
        }

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    showSources(chunks) {
        const sourcesList = document.getElementById('sourcesList');
        sourcesList.innerHTML = '';

        if (!chunks || chunks.length === 0) {
            sourcesList.innerHTML = '<p class="text-muted">暂无相关文档</p>';
            return;
        }

        chunks.forEach((chunk, index) => {
            const sourceItem = document.createElement('div');
            sourceItem.className = 'source-item';
            sourceItem.innerHTML = `
                <small class="text-muted">来源: ${this.escapeHtml(chunk.source)}</small>
                <p class="mb-1">${this.escapeHtml(chunk.content)}</p>
                <small class="text-muted">相关度: ${(1 - chunk.distance).toFixed(2)}</small>
            `;
            sourcesList.appendChild(sourceItem);
        });
    }

    updateStars(rating) {
        const stars = document.querySelectorAll('.star');
        stars.forEach(star => {
            const starRating = parseInt(star.getAttribute('data-rating'));
            if (starRating <= rating) {
                star.classList.add('active');
            } else {
                star.classList.remove('active');
            }
        });
    }

    highlightStars(rating) {
        const stars = document.querySelectorAll('.star');
        stars.forEach(star => {
            const starRating = parseInt(star.getAttribute('data-rating'));
            if (starRating <= rating) {
                star.classList.add('hover');
            } else {
                star.classList.remove('hover');
            }
        });
    }

    async submitRating() {
        if (this.currentRating === 0) {
            alert('请选择评分');
            return;
        }

        try {
            const response = await fetch('/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ rating: this.currentRating })
            });

            const data = await response.json();

            if (!response.ok) {
                this.appendMessage('反馈提交失败: ' + (data.error || '未知错误'), 'system');
                return;
            }

            if (data.error) {
                this.appendMessage('反馈提交失败: ' + data.error, 'system');
            } else {
                this.appendMessage(`感谢您的评分: ${this.currentRating}星`, 'system');
                document.getElementById('ratingSection').style.display = 'none';
            }
        } catch (error) {
            this.appendMessage('反馈提交失败: ' + error.message, 'system');
        }
    }

    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, m => map[m]);
    }

    formatMessage(message) {
        // 简单的格式化，将换行符转换为<br>标签
        return this.escapeHtml(message).replace(/\n/g, '<br>');
    }
}

// 页面加载完成后初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new ChatApp();
});
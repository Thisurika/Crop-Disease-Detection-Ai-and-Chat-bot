document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('fileInput');
    const choosePhotoBtn = document.getElementById('choosePhotoBtn');
    const detectBtn = document.getElementById('detectBtn');
    const resultCard = document.getElementById('resultCard');
    const chatContainer = document.getElementById('chatContainer');
    const chatBody = document.getElementById('chatBody');
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');

    let selectedFile = null;

    // Click "Choose Photo" button
    choosePhotoBtn.addEventListener('click', () => fileInput.click());

    // Click anywhere on upload area (except button) also opens file picker
    uploadArea.addEventListener('click', (e) => {
        if (e.target !== choosePhotoBtn) {
            fileInput.click();
        }
    });

    // Drag & drop
    uploadArea.addEventListener('dragover', e => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));

    uploadArea.addEventListener('drop', e => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handleFile(fileInput.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file (JPG, PNG, WEBP)');
            return;
        }

        selectedFile = file;

        // Show preview
        const reader = new FileReader();
        reader.onload = e => {
            const img = document.createElement('img');
            img.src = e.target.result;
            img.style.maxWidth = '100%';
            img.style.marginTop = '15px';
            img.style.borderRadius = '8px';
            // Remove old preview if any
            const old = uploadArea.querySelector('img.preview');
            if (old) old.remove();
            img.className = 'preview';
            uploadArea.appendChild(img);
            uploadArea.classList.add('preview-visible');
        };
        reader.readAsDataURL(file);

        detectBtn.disabled = false;
    }

    detectBtn.addEventListener('click', async () => {
        if (!selectedFile) return alert('Please select or drop a photo');

        detectBtn.disabled = true;
        detectBtn.textContent = 'Analyzing...';

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            const data = await res.json();

            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }

            // Show result
            document.getElementById('disease-name').textContent = data.disease;
            document.getElementById('confidence-text').textContent = `${data.confidence}%`;
            document.getElementById('confidence-bar').style.width = `${data.confidence}%`;
            document.getElementById('advice-text').textContent = data.advice;
            document.getElementById('preview-img').src = data.image_url;
            document.getElementById('preview-img').style.display = 'block';

            resultCard.style.display = 'block';
            chatContainer.style.display = 'block';

            // Initial chat message
            addMessage('AI', `Detected: ${data.disease} (${data.confidence}%). How can I help with treatment or prevention?`);

        } catch (err) {
            alert('Upload failed: ' + err.message);
        } finally {
            detectBtn.disabled = false;
            detectBtn.textContent = 'Detect Disease';
        }
    });

    // Chat sending
    sendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', e => {
        if (e.key === 'Enter') sendMessage();
    });

    function sendMessage() {
        const text = chatInput.value.trim();
        if (!text) return;

        addMessage('You', text);
        chatInput.value = '';

        fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: text, disease: document.getElementById('disease-name').textContent })
        })
        .then(res => res.json())
        .then(data => addMessage('AI', data.response || 'Sorry, could not get response'))
        .catch(() => addMessage('AI', 'Error connecting to AI'));
    }

    function addMessage(sender, text) {
        const div = document.createElement('div');
        div.className = sender === 'You' ? 'chat-message user-message' : 'chat-message ai-message';
        div.textContent = text;
        chatBody.appendChild(div);
        chatBody.scrollTop = chatBody.scrollHeight;
    }
});